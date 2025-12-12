from unittest.mock import MagicMock

import numpy as np
import pytest

from topicer.embedding.local import LocalEmbedder


@pytest.fixture(scope="function")
def embedder(mocker):
    mock_model = mocker.patch("topicer.embedding.local.SentenceTransformer")
    mock_model_instance = mock_model.return_value

    def mock_encode(text_chunks, *args, **kwargs):
        return np.random.rand(len(text_chunks), 3)

    mock_model_instance.encode = MagicMock(side_effect=mock_encode)

    emb = LocalEmbedder(
        model="prajjwal1/bert-tiny",
        prompt="my prompt",
        query_prompt="my query prompt",
        batch_size=16,
        device="cuda",
        return_fp32=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return emb


def test_embed(embedder):
    res = embedder.embed(["text chunk1", "text chunk2", "text chunk3"], prompt="my prompt", normalize=True)

    assert res.shape == (3, 3)

    embedder.transformer.encode.assert_called_once_with(
        ["text chunk1", "text chunk2", "text chunk3"],
        prompt="my prompt",
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False
    )


def test_embed_queries(embedder):
    res = embedder.embed_queries(["text chunk1", "text chunk2"], normalize=False)
    assert res.shape == (2, 3)
    embedder.transformer.encode.assert_called_once_with(
        ["text chunk1", "text chunk2"],
        prompt="my query prompt",
        normalize_embeddings=False,
        batch_size=16,
        show_progress_bar=False
    )


def test_embed_real_model():
    emb = LocalEmbedder(
        model="prajjwal1/bert-tiny",
        prompt="my prompt",
        query_prompt="my query prompt",
        batch_size=16,
        device=None,
        return_fp32=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    res = emb.embed(["text chunk1", "text chunk2"], prompt="my prompt", normalize=True)
    assert res.shape == (2, 128)
    assert np.allclose(np.linalg.norm(res, ord=2, axis=1), np.array([1., 1.]))

    res = emb.embed("text chunk1", prompt="my prompt", normalize=True)
    assert res.shape == (1, 128)
    assert np.allclose(np.linalg.norm(res, ord=2, axis=1), np.array([1.]))


