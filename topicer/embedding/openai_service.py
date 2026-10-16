import os

import numpy as np
import requests
from classconfig import ConfigurableValue

from topicer.base import BaseEmbeddingService


class OpenAIEmbeddingService(BaseEmbeddingService):
    """Embeddings from an OpenAI-compatible ``/embeddings`` endpoint (e.g. vLLM serving Qwen3-Embedding).

    Queries get ``query_prefix`` (the model's retrieval instruction), documents ``document_prefix``;
    they must match how the stored chunk vectors were embedded.
    """
    base_url: str = ConfigurableValue(
        desc="Base URL of the OpenAI-compatible API, including /v1", user_default="http://qwen-embedder:8000/v1")
    model: str = ConfigurableValue(desc="Served model name", user_default="Qwen/Qwen3-Embedding-0.6B")
    api_key_env: str = ConfigurableValue(
        desc="Environment variable holding the bearer key; unset or empty sends none",
        user_default="EMBEDDING_API_KEY", voluntary=True)
    query_prefix: str = ConfigurableValue(
        desc="Prepended to each query (Qwen3-Embedding: 'Instruct: <task>\\nQuery:')",
        user_default="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        voluntary=True)
    document_prefix: str = ConfigurableValue(desc="Prepended to each document", user_default="", voluntary=True)
    truncate_prompt_tokens: int | None = ConfigurableValue(
        desc="vLLM: truncate longer inputs to this many tokens (-1: the model's maximum) instead of failing",
        user_default=-1, voluntary=True)
    batch_size: int = ConfigurableValue(desc="Texts per request", user_default=64, voluntary=True)
    timeout: int = ConfigurableValue(desc="Timeout of one request in seconds", user_default=30)

    def _embed(self, texts: list[str]) -> np.ndarray:
        url = f"{self.base_url.rstrip('/')}/embeddings"
        key = os.environ.get(self.api_key_env or "", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload = {"model": self.model, "input": batch, "encoding_format": "float"}
            if self.truncate_prompt_tokens is not None:
                payload["truncate_prompt_tokens"] = self.truncate_prompt_tokens
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
            except Exception as e:
                raise ConnectionError(f"Embedding request failed ({url}): {e}")
            if len(data) != len(batch):
                raise ConnectionError(f"Embedding request returned {len(data)} embeddings for {len(batch)} texts ({url})")
            vectors.extend(item["embedding"] for item in data)
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def embed(self, text_chunks: list[str] | str, prompt: str | None = None, normalize: bool | None = True) -> np.ndarray:
        if isinstance(text_chunks, str):
            text_chunks = [text_chunks]
        prefix = self.document_prefix if prompt is None else prompt
        result = self._embed([prefix + text for text in text_chunks])
        return self._normalize(result) if normalize else result

    def embed_queries(self, queries: list[str], normalize: bool | None = None) -> np.ndarray:
        result = self._embed([self.query_prefix + query for query in queries])
        return self._normalize(result) if normalize else result
