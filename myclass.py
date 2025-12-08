import requests
import weaviate
from weaviate.classes.query import MetadataQuery
from typing import List, Dict, Any


class MyClass:
    def __init__(
        self,
        embedding_service_url: str,
        embedding_endpoint: str,
        weaviate_port: int,
        weaviate_grpc_port: int,
        collection_name: str
    ):
        self.embedding_service_url = embedding_service_url
        self.embedding_endpoint = embedding_endpoint
        self.collection_name = collection_name

        # připojení k DB
        try:
            self.client = weaviate.connect_to_local(
                port=weaviate_port,
                grpc_port=weaviate_grpc_port
            )

        except Exception as e:
            raise ConnectionError(f"Nepodařilo se připojit k Weaviate: {e}")

        # kontrola existence kolekce
        if not self.client.collections.exists(self.collection_name):
            self.close()
            raise ValueError(
                f"Kolekce '{self.collection_name}' neexistuje"
            )

    def close(self):
        """Uzavře spojení s databází."""
        if self.client.is_connected():
            self.client.close()

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
                f"Volání embedding služby ({url}) selhalo: {e}")

    def find_relevant_chunks(self, vector: List[float], limit: int = 1000) -> List[Dict[str, Any]]:
        """najde {limit} nejbližších chunků ve Weaviate db k zadanému popisu tagu"""
        try:
            collection = self.client.collections.get(self.collection_name)

            response = collection.query.near_vector(
                near_vector=vector,
                limit=limit,
                return_metadata=MetadataQuery().full(),  # full vraci všechna metadata
            )

            results = []
            for obj in response.objects:
                results.append({
                    "id": str(obj.uuid),
                    "text": obj.properties['text'],
                    "distance": obj.metadata.distance,
                    "metadata": obj.metadata.__dict__
                })

            return results

        except Exception as e:
            print(f"Chyba při dotazu do Weaviate: {e}")
            return []


if __name__ == "__main__":
    import json

    mc = MyClass(
        embedding_service_url="http://localhost:9001",
        embedding_endpoint="/embed_query",
        weaviate_port=9000,
        weaviate_grpc_port=50055,
        collection_name="Chunks"
    )

    try:
        embedding = mc.get_embedding(
            text="Bezpečnost v ulicích, kamery a policejní vyšetřování",
        )
        chunks = mc.find_relevant_chunks(
            vector=embedding,
            limit=1
        )

        for chunk in chunks:
            chunk['text'] = chunk['text'][:30]
            print(json.dumps(chunk, indent=2, default=str, ensure_ascii=False))

    except Exception as e:
        print(f"Chyba: {e}")

    finally:
        mc.close()
