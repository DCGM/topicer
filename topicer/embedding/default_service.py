import requests
import numpy as np

from topicer.base import BaseEmbeddingService
from classconfig import ConfigurableValue


class DefaultEmbeddingService(BaseEmbeddingService):
    service_url: str = ConfigurableValue(
        desc="Service URL", user_default="http://localhost:9001")
    endpoint: str = ConfigurableValue(
        desc="Endpoint", user_default="/embed_query")
    timeout: int = ConfigurableValue(desc="Timeout", user_default=30)

    # https://medium.com/@tellmetiger/embedding-normalization-dummy-learn-2ac8d816e776
    def _normalize(self, embeddings: np.ndarray):
        for i in range(len(embeddings)):
            length = (embeddings[i][0]**2 + embeddings[i][1] **
                      2)**0.5  # Calculate the length
            embeddings[i][0] /= length  # Adjust the x position
            embeddings[i][1] /= length  # Adjust the y position
        return embeddings

    def embed(self, text_chunks: list[str] | str, normalize: bool = True) -> np.ndarray:
        # pokud se jedna o jeden retezec, prevede se na seznam
        if isinstance(text_chunks, str):
            text_chunks = [text_chunks]

        # url
        url = f"{self.service_url.rstrip('/')}{self.endpoint}"

        # odeslani pozadavku
        vectors = []
        for text in text_chunks:
            payload = {"query": text}
            try:
                response = requests.post(
                    url, json=payload, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                vectors.append(data["embedding"])

            except Exception as e:
                raise ConnectionError(f"Embedding failed ({url}): {e}")

        result = np.array(vectors, dtype=np.float32)

        if normalize:
            result = self._normalize(result)

        return result

    def embed_queries(self, text_chunks: list[str] | str, normalize: bool = True) -> np.ndarray:
        return self.embed(text_chunks, normalize)
