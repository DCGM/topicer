import requests
import numpy as np
from topicer.base import BaseEmbeddingService
from classconfig import ConfigurableValue


class DefaultEmbeddingService(BaseEmbeddingService):
    service_url: str = ConfigurableValue(
        desc="Service URL", user_default="http://localhost:9001")
    doc_endpoint: str = ConfigurableValue(
        desc="Endpoint for documents", user_default="/embed_documents")
    query_endpoint: str = ConfigurableValue(
        desc="Endpoint for queries", user_default="/embed_query")
    timeout: int = ConfigurableValue(desc="Timeout", user_default=30)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def embed(self, text_chunks: list[str] | str, prompt: str | None = None, normalize: bool | None = True) -> np.ndarray:
        # pokud se jedna o jeden retezec, prevede se na seznam
        if isinstance(text_chunks, str):
            text_chunks = [text_chunks]

        url = f"{self.service_url.rstrip('/')}{self.doc_endpoint}"

        # DocsRequest(texts: list[str])
        payload = {"texts": text_chunks}

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            result = np.array(data["embeddings"], dtype=np.float32)

        except Exception as e:
            raise ConnectionError(f"Document embedding failed ({url}): {e}")

        if normalize:
            result = self._normalize(result)

        return result

    def embed_queries(self, queries: list[str], normalize: bool | None = None) -> np.ndarray:
        url = f"{self.service_url.rstrip('/')}{self.query_endpoint}"
        vectors = []

        for q in queries:
            # QueryRequest(query: str)
            payload = {"query": q}

            try:
                response = requests.post(
                    url, json=payload, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                vectors.append(data["embedding"])

            except Exception as e:
                raise ConnectionError(f"Query embedding failed for '{q}': {e}")

        result = np.array(vectors, dtype=np.float32)

        if normalize:
            result = self._normalize(result)

        return result
