import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from topicer.llm.api_async import OpenAsyncAPI
from topicer.schemas import DBRequest, Tag, TagProposalUsageSummary, TextChunk, TextChunkWithTagSpanProposals
from topicer.tagging.llm_topicer import LLMTopicer
from topicer.tagging.tagging_schemas import LLMTagProposal, LLMTagProposalList

TAG = Tag(id=uuid4(), name="city")


class FakeDB:
    """Returns the first k candidates in similarity order, like WeaviateService.find_similar_text_chunks."""

    def __init__(self, n_candidates: int, chunks_limit: int = 1000):
        self.chunks_limit = chunks_limit
        self.candidates = [TextChunk(id=uuid4(), text=f"chunk {i} mentions Brno" if i % 2 == 0 else f"chunk {i}")
                           for i in range(n_candidates)]
        self.requested_k = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def find_similar_text_chunks(self, text, embedding, db_request=None, k=None):
        self.requested_k.append(k)
        return self.candidates[:k if k is not None else self.chunks_limit]


class FakeProvider:
    """Fake responses.parse: proposes 'Brno' when present, fails for chunks containing a marker, tracks concurrency."""

    def __init__(self, fail_marker: str | None = None, hang: bool = False):
        self.fail_marker = fail_marker
        self.hang = hang
        self.in_flight = 0
        self.max_in_flight = 0
        self.started = 0
        self.cancelled = 0

    async def parse(self, model, instructions, input, text_format):
        self.in_flight += 1
        self.started += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(3600 if self.hang else 0.001)
            if self.fail_marker and self.fail_marker in input:
                raise RuntimeError("provider error")
            proposals = [LLMTagProposal(tag_id=TAG.id, quote="Brno", confidence=0.9)] if "Brno" in input else []
            return SimpleNamespace(output_parsed=LLMTagProposalList(proposals=proposals),
                                   usage=SimpleNamespace(input_tokens=100, output_tokens=10, cost=0.001))
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_flight -= 1


def make_topicer(db: FakeDB, provider: FakeProvider, **config) -> LLMTopicer:
    with patch("topicer.llm.api_async.AsyncOpenAI") as MockClient:
        llm = OpenAsyncAPI(api_key="dummy", concurrency=100, default_model="fake-model")
    llm.client = MockClient.return_value
    llm.client.responses = MagicMock()
    llm.client.responses.parse = AsyncMock(side_effect=provider.parse)

    topicer = LLMTopicer(**config)
    topicer.set_llm_service(llm)
    topicer.set_db_connection(db)
    embedding = MagicMock()
    embedding.embed_queries.return_value = np.zeros((1, 3))
    topicer.set_embedding_service(embedding)
    return topicer


async def run_stream(topicer: LLMTopicer, max_chunks=None):
    events = [e async for e in topicer.propose_tags_in_db_stream(TAG, DBRequest(), max_chunks=max_chunks)]
    assert isinstance(events[-1], TagProposalUsageSummary)
    assert all(isinstance(e, TextChunkWithTagSpanProposals) for e in events[:-1])
    return events[:-1], events[-1]


async def test_more_candidates_than_limit():
    db, provider = FakeDB(50), FakeProvider()
    results, summary = await run_stream(make_topicer(db, provider), max_chunks=40)

    assert db.requested_k == [41]
    assert provider.started == 40
    assert summary.status == "complete"
    assert summary.tag_id == TAG.id
    assert (summary.max_chunks, summary.max_chunks_source) == (40, "request")
    assert summary.processed_chunks == 40
    assert summary.limit_reached and summary.candidates_truncated
    # Only chunks with proposals are streamed, but usage covers every call including empty results.
    assert len(results) == 20
    assert (summary.usage.attempted_calls, summary.usage.completed_calls, summary.usage.failed_calls) == (40, 40, 0)
    assert all(c.cost_usd == 0.001 and c.cost_source == "reported" for c in summary.usage.calls)


async def test_processes_most_similar_candidates():
    db, provider = FakeDB(50), FakeProvider()
    results, _ = await run_stream(make_topicer(db, provider), max_chunks=10)
    assert {r.id for r in results} == {c.id for c in db.candidates[:10] if "Brno" in c.text}


async def test_exactly_limit_candidates():
    db, provider = FakeDB(40), FakeProvider()
    _, summary = await run_stream(make_topicer(db, provider), max_chunks=40)
    assert summary.processed_chunks == 40
    assert summary.limit_reached
    assert not summary.candidates_truncated


async def test_fewer_candidates_than_limit():
    db, provider = FakeDB(5), FakeProvider()
    _, summary = await run_stream(make_topicer(db, provider), max_chunks=40)
    assert summary.processed_chunks == 5
    assert not summary.limit_reached
    assert not summary.candidates_truncated
    assert summary.usage.attempted_calls == 5


async def test_omitted_max_chunks_uses_config_limit():
    db, provider = FakeDB(30, chunks_limit=25), FakeProvider()
    _, summary = await run_stream(make_topicer(db, provider))
    assert db.requested_k == [26]
    assert (summary.max_chunks, summary.max_chunks_source) == (25, "config")
    assert summary.processed_chunks == 25
    assert summary.limit_reached and summary.candidates_truncated


async def test_no_candidates_still_sends_summary():
    db, provider = FakeDB(0), FakeProvider()
    results, summary = await run_stream(make_topicer(db, provider), max_chunks=40)
    assert results == []
    assert summary.status == "complete"
    assert summary.processed_chunks == 0
    assert summary.usage.attempted_calls == 0


async def test_no_proposals_still_accounts_usage():
    db, provider = FakeDB(0), FakeProvider()
    db.candidates = [TextChunk(id=uuid4(), text="nothing here") for _ in range(3)]
    results, summary = await run_stream(make_topicer(db, provider), max_chunks=40)
    assert results == []
    assert summary.processed_chunks == 3
    assert summary.usage.completed_calls == 3


@pytest.mark.parametrize("max_chunks", [0, -1, 201, True, 2.5])
async def test_invalid_max_chunks_rejected_before_inference(max_chunks):
    db, provider = FakeDB(10), FakeProvider()
    topicer = make_topicer(db, provider)
    with pytest.raises(ValueError):
        topicer.validate_max_chunks(max_chunks)
    with pytest.raises(ValueError):
        await run_stream(topicer, max_chunks=max_chunks)
    assert db.requested_k == [] and provider.started == 0


async def test_configurable_ceiling():
    topicer = make_topicer(FakeDB(1), FakeProvider(), max_chunks_ceiling=40)
    topicer.validate_max_chunks(40)
    with pytest.raises(ValueError):
        topicer.validate_max_chunks(41)


async def test_partial_failures_are_reported():
    db, provider = FakeDB(6), FakeProvider(fail_marker="chunk 1")
    results, summary = await run_stream(make_topicer(db, provider), max_chunks=40)
    assert summary.status == "partial"
    assert summary.processed_chunks == 6
    assert len(results) == 3
    usage = summary.usage
    assert (usage.attempted_calls, usage.completed_calls, usage.failed_calls) == (6, 5, 1)
    failed = [c for c in usage.calls if c.outcome == "error"]
    assert failed[0].cost_usd is None and failed[0].cost_source == "unknown"


async def test_all_failures_are_failed_status():
    db, provider = FakeDB(3), FakeProvider(fail_marker="chunk")
    _, summary = await run_stream(make_topicer(db, provider), max_chunks=40)
    assert summary.status == "failed"
    assert summary.usage.failed_calls == 3


async def test_retrieval_failure_sends_failed_summary():
    db, provider = FakeDB(3), FakeProvider()
    db.find_similar_text_chunks = AsyncMock(side_effect=RuntimeError("weaviate down"))
    results, summary = await run_stream(make_topicer(db, provider), max_chunks=40)
    assert results == []
    assert summary.status == "failed"
    assert summary.processed_chunks == 0


async def test_in_flight_chunks_are_bounded():
    db, provider = FakeDB(30), FakeProvider()
    _, summary = await run_stream(make_topicer(db, provider, max_concurrent_chunks=4), max_chunks=30)
    assert summary.processed_chunks == 30
    assert provider.max_in_flight == 4


async def test_closing_stream_cancels_outstanding_calls(caplog):
    db, provider = FakeDB(10), FakeProvider(hang=True)
    topicer = make_topicer(db, provider, max_concurrent_chunks=3)
    stream = topicer.propose_tags_in_db_stream(TAG, DBRequest(), max_chunks=10)
    consumer = asyncio.create_task(stream.__anext__())
    while provider.started < 3:
        await asyncio.sleep(0.001)

    # Client disconnect: Starlette cancels the task iterating the stream.
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert provider.started == 3
    assert provider.cancelled == 3
    assert provider.in_flight == 0
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert leaked == []
    assert "closed before its usage summary was delivered" in caplog.text
    assert "attempted_calls=3" in caplog.text


SECRET = "SECRET document text and prompt"


async def test_chunk_failure_log_has_no_exception_message(caplog):
    db, provider = FakeDB(2), FakeProvider()
    topicer = make_topicer(db, provider)
    topicer.llm_service.client.responses.parse.side_effect = RuntimeError(SECRET)
    _, summary = await run_stream(topicer, max_chunks=40)

    assert summary.status == "failed"
    assert "RuntimeError" in caplog.text and "llm_inference" in caplog.text
    assert SECRET not in caplog.text


async def test_retrieval_failure_log_has_no_exception_message(caplog):
    db, provider = FakeDB(2), FakeProvider()
    db.find_similar_text_chunks = AsyncMock(side_effect=RuntimeError(SECRET))
    _, summary = await run_stream(make_topicer(db, provider), max_chunks=40)

    assert summary.status == "failed"
    assert "RuntimeError" in caplog.text and "retrieval" in caplog.text
    assert SECRET not in caplog.text
