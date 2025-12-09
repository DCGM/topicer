import requests
from typing import List


class Embedder:
    def __init__(
        self,
        embedding_service_url: str,
        embedding_endpoint: str,
        collection_name: str
    ):
        self.embedding_service_url = embedding_service_url
        self.embedding_endpoint = embedding_endpoint
        self.collection_name = collection_name

    def get_embedding(self, text: str) -> List[float]:
        """Získá vektor voláním BE služby."""
        url = f"{self.embedding_service_url}{self.embedding_endpoint}"
        payload = {"query": text}

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()  # při odpovědi 4xx nebo 5xx vyvolá výjimku

            data = response.json()
            return data["embedding"]

        except Exception as e:
            raise ConnectionError(
                f"Volání embedding služby {url} selhalo: {e}")


if __name__ == "__main__":
    import json

    embedding_service = Embedder(
        embedding_service_url="http://localhost:9001",
        embedding_endpoint="/embed_query",
        collection_name="Chunks"
    )

    try:
        embedding = embedding_service.get_embedding(
            text="Bezpečnost v ulicích, kamery a policejní vyšetřování",
        )

        print(json.dumps(embedding[:20], indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Chyba: {e}")
