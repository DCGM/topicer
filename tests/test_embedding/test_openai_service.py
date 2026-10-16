from unittest.mock import MagicMock

import numpy as np
import pytest

from topicer.embedding.openai_service import OpenAIEmbeddingService

QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


def service(**config):
    return OpenAIEmbeddingService(**{"base_url": "http://qwen-embedder:8000/v1/", "model": "Qwen/Qwen3-Embedding-0.6B",
                                     "timeout": 5, "batch_size": 2, **config})


@pytest.fixture
def post(mocker):
    """requests.post answering each input with [len(text), 0] in reversed order (the index orders them)."""
    def answer(url, json, headers, timeout):
        response = MagicMock()
        response.json.return_value = {"data": [{"index": i, "embedding": [float(len(t)), 0.0]}
                                               for i, t in reversed(list(enumerate(json["input"])))]}
        return response
    return mocker.patch("topicer.embedding.openai_service.requests.post", side_effect=answer)


@pytest.mark.unit
def test_queries_get_the_query_instruction_and_are_batched(post, monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    result = service().embed_queries(["a", "bb", "ccc"])

    assert [call.kwargs["json"]["input"] for call in post.call_args_list] == [
        [QUERY_PREFIX + "a", QUERY_PREFIX + "bb"], [QUERY_PREFIX + "ccc"]]
    first = post.call_args_list[0].kwargs
    assert first["json"]["model"] == "Qwen/Qwen3-Embedding-0.6B" and first["json"]["truncate_prompt_tokens"] == -1
    assert post.call_args_list[0].args[0] == "http://qwen-embedder:8000/v1/embeddings" and first["headers"] == {}
    np.testing.assert_array_equal(result[:, 0], [len(QUERY_PREFIX) + n for n in (1, 2, 3)])


@pytest.mark.unit
def test_documents_get_the_document_prefix_normalized_and_the_key(post, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")

    result = service(document_prefix="passage: ").embed("text")

    assert post.call_args.kwargs["json"]["input"] == ["passage: text"]
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer secret"}
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0])


@pytest.mark.unit
def test_a_failed_request_raises_connection_error(mocker):
    mocker.patch("topicer.embedding.openai_service.requests.post", side_effect=RuntimeError("down"))

    with pytest.raises(ConnectionError, match="Embedding request failed"):
        service().embed_queries(["a"])
