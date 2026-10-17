import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError
from pydantic import BaseModel

from topicer.llm.api_async import OllamaAsyncAPI, OpenAsyncAPI
from topicer.llm.usage import collect_usage, estimate_cost, meter_call, openai_usage

PRICES = {"priced-model": {"input_per_million": 1.0, "output_per_million": 4.0}}


class OutputModel(BaseModel):
    summary: str


def openai_response(input_tokens=None, output_tokens=None, cost=None, parsed=None, text=None):
    usage = None
    if input_tokens is not None or output_tokens is not None or cost is not None:
        usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
        if cost is not None:
            usage.cost = cost
    return SimpleNamespace(output_parsed=parsed, output_text=text, usage=usage)


@pytest.fixture
def open_api():
    with patch("topicer.llm.api_async.AsyncOpenAI") as MockClient:
        instance = OpenAsyncAPI(api_key="dummy", base_url="http://dummy", concurrency=2, pool_interval=0,
                                prices=PRICES)
        instance.client = MockClient.return_value
        instance.client.responses = MagicMock()
        instance.client.responses.create = AsyncMock()
        instance.client.responses.parse = AsyncMock()
        return instance


async def run_parse(api, model):
    return await api.process_single_request("text", "instruction", OutputModel, model)


def test_estimate_cost():
    assert estimate_cost("priced-model", 1_000_000, 500_000, PRICES) == pytest.approx(3.0)
    assert estimate_cost("other-model", 10, 10, PRICES) is None
    assert estimate_cost("priced-model", None, 10, PRICES) is None
    assert estimate_cost("priced-model", 10, 10, None) is None


def test_openai_usage_chat_completions_and_extra_cost():
    usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "model_extra"])
    usage.prompt_tokens, usage.completion_tokens, usage.model_extra = 7, 3, {"cost": 0.5}
    assert openai_usage(SimpleNamespace(usage=usage)) == (7, 3, 0.5)
    assert openai_usage(SimpleNamespace(usage=None)) == (None, None, None)


async def test_reported_cost(open_api):
    open_api.client.responses.parse.return_value = openai_response(1200, 160, cost=0.002,
                                                                   parsed=OutputModel(summary="x"))
    with collect_usage() as collector:
        result = await run_parse(open_api, "priced-model")

    assert result.summary == "x"
    usage = collector.summary()
    assert (usage.attempted_calls, usage.completed_calls, usage.failed_calls) == (1, 1, 0)
    call = usage.calls[0]
    assert call.model_dump() == {
        "provider": "openai", "model": "priced-model", "outcome": "success",
        "input_tokens": 1200, "output_tokens": 160, "usage_source": "reported",
        "cost_usd": 0.002, "cost_source": "reported",
    }


async def test_estimated_cost(open_api):
    open_api.client.responses.parse.return_value = openai_response(1_000_000, 1_000_000,
                                                                   parsed=OutputModel(summary="x"))
    with collect_usage() as collector:
        await run_parse(open_api, "priced-model")

    call = collector.calls[0]
    assert call.cost_usd == pytest.approx(5.0)
    assert call.cost_source == "estimated"


async def test_unknown_cost_is_not_zero(open_api):
    open_api.client.responses.parse.return_value = openai_response(10, 20, parsed=OutputModel(summary="x"))
    with collect_usage() as collector:
        await run_parse(open_api, "unpriced-model")

    call = collector.calls[0]
    assert (call.input_tokens, call.output_tokens, call.usage_source) == (10, 20, "reported")
    assert call.cost_usd is None
    assert call.cost_source == "unknown"


async def test_missing_usage_is_unknown(open_api):
    open_api.client.responses.parse.return_value = openai_response(parsed=OutputModel(summary="x"))
    with collect_usage() as collector:
        await run_parse(open_api, "priced-model")

    call = collector.calls[0]
    assert call.outcome == "success"
    assert (call.input_tokens, call.output_tokens, call.usage_source) == (None, None, "unknown")
    assert (call.cost_usd, call.cost_source) == (None, "unknown")


async def test_failed_call_recorded_as_unknown(open_api):
    open_api.client.responses.parse.side_effect = ValueError("could not parse")
    with collect_usage() as collector:
        with pytest.raises(ValueError):
            await run_parse(open_api, "priced-model")

    usage = collector.summary()
    assert (usage.attempted_calls, usage.completed_calls, usage.failed_calls) == (1, 0, 1)
    call = usage.calls[0]
    assert call.outcome == "error"
    assert (call.cost_usd, call.cost_source, call.input_tokens) == (None, "unknown", None)


async def test_rate_limit_retries_are_separate_attempts(open_api):
    open_api.client.responses.parse.side_effect = [
        RateLimitError(message="Rate limit", response=MagicMock(), body=None),
        openai_response(5, 6, cost=0.1, parsed=OutputModel(summary="x")),
    ]
    with collect_usage() as collector:
        await run_parse(open_api, "priced-model")

    assert [c.outcome for c in collector.calls] == ["error", "success"]
    assert collector.summary().attempted_calls == 2


async def test_cancelled_call_recorded(open_api):
    started = asyncio.Event()

    async def hang(**kwargs):
        started.set()
        await asyncio.sleep(10)

    open_api.client.responses.parse.side_effect = hang
    with collect_usage() as collector:
        task = asyncio.create_task(run_parse(open_api, "priced-model"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert [c.outcome for c in collector.calls] == ["cancelled"]
    assert collector.calls[0].cost_source == "unknown"


async def test_tasks_inherit_collector_and_no_collector_is_noop(open_api):
    open_api.client.responses.parse.return_value = openai_response(1, 1, parsed=OutputModel(summary="x"))
    with collect_usage() as collector:
        await open_api.process_text_chunks_structured(["a", "b", "c"], "instruction", OutputModel, "m")
    assert collector.summary().completed_calls == 3

    # Outside a collector the calls are not recorded anywhere and nothing fails.
    await run_parse(open_api, "m")
    assert len(collector.calls) == 3


async def test_configured_provider_name(open_api):
    open_api.provider = "openrouter"
    open_api.client.responses.parse.return_value = openai_response(1, 1, parsed=OutputModel(summary="x"))
    with collect_usage() as collector:
        await run_parse(open_api, "m")
    assert collector.calls[0].provider == "openrouter"


async def test_ollama_token_counts():
    with patch("topicer.llm.api_async.AsyncClient") as MockClient:
        api = OllamaAsyncAPI(api_key="dummy", base_url="http://localhost:11434", concurrency=2,
                             prices={"llama": {"input_per_million": 0, "output_per_million": 0}})
        api.client = MockClient.return_value
        api.client.chat = AsyncMock(return_value=SimpleNamespace(
            message=SimpleNamespace(content="hello"), prompt_eval_count=11, eval_count=4))

        with collect_usage() as collector:
            assert await api.process_single_request("text", "instruction", None, "llama") == "hello"

    call = collector.calls[0]
    assert (call.provider, call.input_tokens, call.output_tokens) == ("ollama", 11, 4)
    assert (call.cost_usd, call.cost_source) == (0.0, "estimated")


def test_meter_without_success_report_is_unknown():
    with collect_usage() as collector:
        with meter_call("p", "m"):
            pass
    assert collector.calls[0].outcome == "success"
    assert collector.calls[0].usage_source == "unknown"


def rate_limit_error():
    return RateLimitError(message="Rate limit", response=MagicMock(status_code=429), body=None)


def test_default_rate_limit_retry_interval():
    with patch("topicer.llm.api_async.AsyncOpenAI"):
        api = OpenAsyncAPI(api_key="dummy")
    assert api.pool_interval == 5


async def test_persistent_rate_limit_gives_up_after_retry_budget(open_api):
    open_api.rate_limit_retries = 2
    open_api.client.responses.parse.side_effect = rate_limit_error()
    with collect_usage() as collector:
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RateLimitError):
                await run_parse(open_api, "priced-model")

    assert open_api.client.responses.parse.call_count == 3
    assert mock_sleep.call_count == 2
    usage = collector.summary()
    assert (usage.attempted_calls, usage.failed_calls) == (3, 3)
    assert all(c.outcome == "error" and c.cost_source == "unknown" for c in usage.calls)


async def test_zero_retries_raises_immediately(open_api):
    open_api.rate_limit_retries = 0
    open_api.client.responses.parse.side_effect = rate_limit_error()
    with pytest.raises(RateLimitError):
        await run_parse(open_api, "priced-model")
    assert open_api.client.responses.parse.call_count == 1


async def test_cancellation_during_rate_limit_backoff(open_api):
    open_api.pool_interval = 3600
    open_api.client.responses.parse.side_effect = rate_limit_error()
    with collect_usage() as collector:
        task = asyncio.create_task(run_parse(open_api, "priced-model"))
        while not collector.calls:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)

    # The failed attempt stays metered; no further attempt was started.
    assert [c.outcome for c in collector.calls] == ["error"]
    assert open_api.client.responses.parse.call_count == 1


async def test_openai_service_estimates_cost_from_prices():
    from topicer.llm.openai import OpenAIService

    service = OpenAIService(prices=PRICES, model="priced-model")
    service._client = MagicMock()
    service._client.responses.parse = AsyncMock(
        return_value=openai_response(1_000_000, 1_000_000, parsed=OutputModel(summary="x")))
    with collect_usage() as collector:
        await service.process_text_chunks_structured(["a"], "instruction", OutputModel)

    call = collector.calls[0]
    assert (call.cost_usd, call.cost_source) == (pytest.approx(5.0), "estimated")


def test_describe_exception_omits_message():
    from topicer.llm.usage import describe_exception

    assert describe_exception(ValueError("secret document text")) == "ValueError"
    assert describe_exception(rate_limit_error()) == "RateLimitError (status 429)"
