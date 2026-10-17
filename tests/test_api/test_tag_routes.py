import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from topicer_api.main import app
from topicer_api.topicers import get_loaded_topicers
from tests.test_tagging.test_llm_topicer_stream import TAG, FakeDB, FakeProvider, make_topicer


@pytest.fixture
def setup():
    db, provider = FakeDB(50), FakeProvider()
    topicer = make_topicer(db, provider)
    app.dependency_overrides[get_loaded_topicers] = lambda: {"llm": topicer}
    # Without a `with` block TestClient does not run the lifespan, so no configs are loaded from disk.
    yield TestClient(app), db, provider
    app.dependency_overrides.clear()


def stream_body():
    return {"tag": TAG.model_dump(mode="json"), "db_request": {"collection_id": str(uuid4())}}


def stream_lines(client, **params):
    response = client.post("/v1/tags/propose/db/stream", params={"config_name": "llm", **params}, json=stream_body())
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_stream_ends_with_usage_summary(setup):
    client, db, provider = setup
    lines = stream_lines(client, max_chunks=40)

    *results, summary = lines
    assert len(results) == 20
    for r in results:
        assert "event" not in r
        assert {"id", "text", "tag_span_proposals"} <= r.keys()
    assert summary["event"] == "usage_summary"
    assert summary["status"] == "complete"
    assert summary["tag_id"] == str(TAG.id)
    assert (summary["max_chunks"], summary["processed_chunks"]) == (40, 40)
    assert summary["limit_reached"] is True and summary["candidates_truncated"] is True
    usage = summary["usage"]
    assert (usage["attempted_calls"], usage["completed_calls"], usage["failed_calls"]) == (40, 40, 0)
    assert len(usage["calls"]) == 40
    assert usage["calls"][0] == {
        "provider": "openai", "model": "fake-model", "outcome": "success",
        "input_tokens": 100, "output_tokens": 10, "usage_source": "reported",
        "cost_usd": 0.001, "cost_source": "reported",
    }
    assert db.requested_k == [41]


def test_stream_without_max_chunks_uses_config_limit(setup):
    client, db, provider = setup
    summary = stream_lines(client)[-1]
    assert (summary["max_chunks"], summary["max_chunks_source"]) == (1000, "config")
    assert summary["processed_chunks"] == 50
    assert summary["limit_reached"] is False and summary["candidates_truncated"] is False


@pytest.mark.parametrize("max_chunks", ["0", "-5", "abc", "1.5", "201"])
def test_stream_invalid_max_chunks_is_422_before_inference(setup, max_chunks):
    client, db, provider = setup
    response = client.post("/v1/tags/propose/db/stream", params={"config_name": "llm", "max_chunks": max_chunks},
                           json=stream_body())
    assert response.status_code == 422
    assert db.requested_k == [] and provider.started == 0


def test_propose_texts_returns_usage(setup):
    client, db, provider = setup
    chunk = {"id": str(uuid4()), "text": "We met in Brno."}
    response = client.post("/v1/tags/propose/texts", params={"config_name": "llm"},
                           json={"text_chunk": chunk, "tags": [TAG.model_dump(mode="json")]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == chunk["id"] and body["text"] == chunk["text"]
    assert len(body["tag_span_proposals"]) == 1
    assert body["usage"]["attempted_calls"] == 1
    assert body["usage"]["calls"][0]["cost_usd"] == 0.001


def test_propose_texts_has_no_max_chunks_param(setup):
    # max_chunks is not part of /texts; unknown query params are ignored and usage is still reported.
    client, db, provider = setup
    response = client.post("/v1/tags/propose/texts", params={"config_name": "llm", "max_chunks": 1},
                           json={"text_chunk": {"id": str(uuid4()), "text": "x"}, "tags": [TAG.model_dump(mode="json")]})
    assert response.status_code == 200
    assert "max_chunks" not in [p["name"] for p in
                                app.openapi()["paths"]["/v1/tags/propose/texts"]["post"].get("parameters", [])]


def test_openapi_documents_stream_union():
    schema = app.openapi()
    operation = schema["paths"]["/v1/tags/propose/db/stream"]["post"]
    line_schema = operation["responses"]["200"]["content"]["application/x-ndjson"]["schema"]
    refs = {s["$ref"].rsplit("/", 1)[-1] for s in line_schema["oneOf"]}
    assert refs == {"TextChunkWithTagSpanProposals", "TagProposalUsageSummary"}
    for ref in refs | {"LLMUsage", "LLMCallUsage"}:
        assert ref in schema["components"]["schemas"]
    assert "max_chunks" in [p["name"] for p in operation["parameters"]]
    texts_schema = schema["components"]["schemas"]["TextChunkWithTagSpanProposalsAndUsage"]
    assert "usage" in texts_schema["properties"] and "usage" not in texts_schema.get("required", [])
