from topicer.base import BaseDBConnection
from classconfig import ConfigurableMixin, ConfigurableValue
import weaviate
from weaviate import WeaviateClient
from weaviate.classes.query import Filter
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

        # Definition of the filter using reference property
        chunk_filter = Filter.by_ref(self.chunk_user_collection_ref).by_id().equal(db_request.collection_id) if (
            db_request.collection_id is not None
        ) else None

        MAX_TOTAL_LIMIT = max(100000, self.chunks_limit)
        results: list[TextChunk] = []
        cursor = None
        batch_size = 1000  # Number of results to fetch per request

        while len(results) < MAX_TOTAL_LIMIT:
            response = chunks_collection.query.fetch_objects(
                filters=chunk_filter,
                limit=min(batch_size, MAX_TOTAL_LIMIT - len(results)),
                after=cursor,
                return_properties=[self.chunk_text_prop],
            )

            if not response.objects:
                break  # No more results

            for obj in response.objects:
                results.append(
                    TextChunk(
                        id=obj.uuid,
                        text=obj.properties.get(self.chunk_text_prop, ""),
                    )
                )

            # Update cursor to the last fetched object's UUID
            cursor = response.objects[-1].uuid

        # Return all fetched results in a single list
        return results

    # TODO: Discuss whether this streaming approach is better than the above method
    def get_text_chunks_stream(self, db_request: DBRequest) -> Iterable[TextChunk]:
        chunks_collection = self._client.collections.use(
            self.chunks_collection)

        chunk_filter = Filter.by_ref(self.chunk_user_collection_ref).by_id().equal(db_request.collection_id) if (
            db_request.collection_id is not None
        ) else None
        
        MAX_TOTAL_LIMIT = max(100000, self.chunks_limit)
        results_fetched = 0
        cursor = None
        batch_size = 1000  # Number of results to fetch per request
        
        while results_fetched < MAX_TOTAL_LIMIT:
            response = chunks_collection.query.fetch_objects(
                filters=chunk_filter,
                limit=min(batch_size, MAX_TOTAL_LIMIT - results_fetched),
                after=cursor,
                return_properties=[self.chunk_text_prop],
            )
            
            if not response.objects:
                break  # No more results

            for obj in response.objects:
                yield TextChunk(
                    id=obj.uuid,
                    text=obj.properties.get(self.chunk_text_prop, ""),
                )
                results_fetched += 1
            
            # Update cursor to the last fetched object's UUID
            cursor = response.objects[-1].uuid

    def find_similar_text_chunks(
        self,
        text: str,
        embedding: np.ndarray,
        db_request: DBRequest | None = None,
        k: int | None = None,
    ) -> list[TextChunk]:
        chunks_coll_name = self.chunks_collection
        where_filter = None
        if (
            db_request
            and db_request.collection_id is not None
            and self.chunk_user_collection_id_prop
        ):
            where_filter = Filter.by_property(self.chunk_user_collection_id_prop).equal(
                str(db_request.collection_id)
            )

        top_k = k if k is not None else self.chunks_limit
        vec = embedding.tolist()

        chunks_collection = self._client.collections.use(chunks_coll_name)
        response = chunks_collection.query.hybrid(
            vector=vec,
            alpha=self.hybrid_search_alpha,
            filters=where_filter,
            return_properties=[self.chunk_text_prop],
            limit=top_k,
        )
