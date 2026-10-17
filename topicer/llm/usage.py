"""
Metering of underlying LLM provider calls.

LLM adapters wrap every provider attempt in :func:`meter_call`, which records an :class:`LLMCallUsage` into the
collector active in the current context (if any). Callers that need usage open a collector with
:func:`collect_usage` around their work; asyncio tasks created inside inherit it.

Only counts, model and provider are recorded: no prompts, texts or credentials.
"""
import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from topicer.schemas import LLMCallUsage, LLMUsage

logger = logging.getLogger(__name__)

_current_collector: ContextVar[Optional["UsageCollector"]] = ContextVar("topicer_llm_usage_collector", default=None)


class UsageCollector:
    """Accumulates the usage of all LLM calls made while it is active."""

    def __init__(self) -> None:
        self.calls: list[LLMCallUsage] = []

    def record(self, call: LLMCallUsage) -> None:
        self.calls.append(call)

    def summary(self) -> LLMUsage:
        return LLMUsage.from_calls(self.calls)


@contextmanager
def collect_usage(collector: UsageCollector | None = None) -> Iterator[UsageCollector]:
    """Make `collector` (a new one when None) receive the usage of LLM calls made in this context."""
    collector = collector if collector is not None else UsageCollector()
    token = _current_collector.set(collector)
    try:
        yield collector
    finally:
        _current_collector.reset(token)


def record_call(call: LLMCallUsage) -> None:
    collector = _current_collector.get()
    if collector is not None:
        collector.record(call)


def describe_exception(exc: BaseException) -> str:
    """
    Safe description of an exception for logs: its class and, for provider errors, the HTTP status code.

    The message is left out because provider errors may embed prompts, response bodies or document text.
    """
    description = type(exc).__name__
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        description += f" (status {status_code})"
    return description


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def estimate_cost(model: str | None, input_tokens: int | None, output_tokens: int | None,
                  prices: dict | None) -> float | None:
    """
    Estimate a call's cost from the server-configured prices.

    :param prices: {model: {"input_per_million": USD, "output_per_million": USD}}
    :return: cost in USD, or None when the model has no price or the token counts are unknown.
    """
    if not prices or model is None or input_tokens is None or output_tokens is None:
        return None
    price = prices.get(model)
    if not price:
        return None
    try:
        return (input_tokens * float(price["input_per_million"]) +
                output_tokens * float(price["output_per_million"])) / 1_000_000
    except (KeyError, TypeError, ValueError):
        logger.warning(f"Invalid price configuration for model '{model}'.")
        return None


def openai_usage(response: Any) -> tuple[int | None, int | None, float | None]:
    """
    Extract (input_tokens, output_tokens, reported cost) from an OpenAI(-compatible) response.

    Handles the Responses API (input_tokens/output_tokens) and Chat Completions (prompt_tokens/completion_tokens).
    The cost is only present with providers that report it, e.g. OpenRouter's `usage.cost`.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    def field(name: str) -> Any:
        value = getattr(usage, name, None)
        if value is None and isinstance(getattr(usage, "model_extra", None), dict):
            value = usage.model_extra.get(name)
        return value

    input_tokens = _int_or_none(field("input_tokens"))
    if input_tokens is None:
        input_tokens = _int_or_none(field("prompt_tokens"))
    output_tokens = _int_or_none(field("output_tokens"))
    if output_tokens is None:
        output_tokens = _int_or_none(field("completion_tokens"))
    return input_tokens, output_tokens, _float_or_none(field("cost"))


class CallMeter:
    """Handle of a metered call; the adapter reports the provider's usage with :meth:`success`."""

    def __init__(self, provider: str, model: str | None, prices: dict | None) -> None:
        self.provider = provider
        self.model = model
        self.prices = prices
        self.recorded = False

    def success(self, input_tokens: Any = None, output_tokens: Any = None, reported_cost: Any = None) -> None:
        input_tokens = _int_or_none(input_tokens)
        output_tokens = _int_or_none(output_tokens)
        cost = _float_or_none(reported_cost)
        cost_source = "reported"
        if cost is None:
            cost = estimate_cost(self.model, input_tokens, output_tokens, self.prices)
            cost_source = "estimated" if cost is not None else "unknown"

        self._record(LLMCallUsage(
            provider=self.provider, model=self.model, outcome="success",
            input_tokens=input_tokens, output_tokens=output_tokens,
            usage_source="reported" if input_tokens is not None and output_tokens is not None else "unknown",
            cost_usd=cost, cost_source=cost_source,
        ))

    def failure(self, outcome: str) -> None:
        self._record(LLMCallUsage(provider=self.provider, model=self.model, outcome=outcome))

    def _record(self, call: LLMCallUsage) -> None:
        if not self.recorded:
            self.recorded = True
            record_call(call)


@contextmanager
def meter_call(provider: str, model: str | None, prices: dict | None = None) -> Iterator[CallMeter]:
    """
    Meter one provider attempt. Call `meter.success(...)` with the reported usage once the response arrives.

    An exception or cancellation records the attempt with unknown usage and cost (it may have been billed);
    a response without a `success` report is recorded as successful with unknown usage.
    """
    meter = CallMeter(provider, model, prices)
    try:
        yield meter
    except asyncio.CancelledError:
        meter.failure("cancelled")
        raise
    except BaseException:
        meter.failure("error")
        raise
    else:
        meter.success()
