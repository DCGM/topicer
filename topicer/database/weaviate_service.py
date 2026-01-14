from topicer.base import BaseDBConnection
from classconfig import ConfigurableMixin, ConfigurableValue
import weaviate
from weaviate import WeaviateClient
from weaviate.classes.query import Filter, Sort
from topicer.schemas import TextChunk, DBRequest
from uuid import UUID
import numpy as np
from itertools import islice
from collections.abc import Iterable


class WeaviateService(BaseDBConnection, ConfigurableMixin):
    # Connection config
    host: str = ConfigurableValue(
        desc="Weaviate host", user_default="localhost")
    rest_port: int = ConfigurableValue(
        desc="Weaviate REST port", user_default=8080)
    grpc_port: int = ConfigurableValue(
        desc="Weaviate gRPC port", user_default=50051)

    # Data model config
    chunks_collection: str = ConfigurableValue(
        desc="Collection/class name storing text chunks",
        user_default="Chunks_test",
    )
    # Property on chunk objects that links/filters by user collection id
    chunk_user_collection_ref: str = ConfigurableValue(
        desc="Property on Chunks referencing the user collection",
        user_default="userCollection",
        voluntary=True,
    )
    # Property holding the text in chunk objects
    chunk_text_prop: str = ConfigurableValue(
        desc="Property name of the text field within the chunks collection",
        user_default="text",
        voluntary=True,
    )

    chunks_limit: int = ConfigurableValue(
        desc="Max number of chunks to retrieve per request",
        user_default=100000,
    )

    hybrid_search_alpha: float = ConfigurableValue(
        desc="Alpha parameter for hybrid search (0.0 = pure keyword search, 1.0 = pure vector search)",
        user_default=0.5,
    )

    def __post_init__(self):
        self._client: WeaviateClient = weaviate.connect_to_custom(
            http_host=self.host,
            http_port=self.rest_port,
            http_secure=False,
            grpc_host=self.host,
            grpc_port=self.grpc_port,
            grpc_secure=False,
        )

    def get_text_chunks(self, db_request: DBRequest) -> list[TextChunk]:
        # Access the chunks collection
        chunks_collection = self._client.collections.use(
            self.chunks_collection)

        results: list[TextChunk] = []
        MAX_TOTAL_LIMIT = max(100000, self.chunks_limit)

        # Definition of the filter using reference property
        chunk_filter = Filter.by_ref(self.chunk_user_collection_ref).by_id().equal(db_request.collection_id) if (
            db_request.collection_id is not None
        ) else None

        response = chunks_collection.query.fetch_objects(
            filters=chunk_filter,
            limit=MAX_TOTAL_LIMIT,
            return_properties=[self.chunk_text_prop],
        )

        for obj in response.objects:
            results.append(
                TextChunk(
                    id=obj.uuid,
                    text=obj.properties.get(self.chunk_text_prop, ""),
                )
            )

        # Return all fetched results in a single list
        return results

    def find_similar_text_chunks(
        self,
        text: str,
        embedding: np.ndarray,
        db_request: DBRequest | None = None,
        k: int | None = None,
    ) -> list[TextChunk]:
        # Access the chunks collection
        chunks_collection = self._client.collections.use(
            self.chunks_collection)

        results: list[TextChunk] = []
        top_k = k if k is not None else self.chunks_limit

        chunk_filter = Filter.by_ref(
            self.chunk_user_collection_ref).by_id().equal(db_request.collection_id) if (
                db_request is not None and db_request.collection_id is not None
        ) else None

        vec = embedding.tolist()

        response = chunks_collection.query.hybrid(
            query=text,
            vector=vec,
            alpha=self.hybrid_search_alpha,
            filters=chunk_filter,
            return_properties=[self.chunk_text_prop],
            limit=top_k,
        )
        
        for obj in response.objects:
            results.append(
                TextChunk(
                    id=obj.uuid,
                    text=obj.properties.get(self.chunk_text_prop, ""),
                )
            )
        
        # Return the results
        return results
              

    def get_embeddings(self, text_chunks: list[TextChunk]) -> np.ndarray:
        pass
