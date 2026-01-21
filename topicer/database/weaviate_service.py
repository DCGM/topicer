from topicer.base import BaseDBConnection
from classconfig import ConfigurableMixin, ConfigurableValue
import weaviate
from weaviate import WeaviateAsyncClient
from weaviate.classes.query import Filter
from topicer.schemas import TextChunk, DBRequest
import numpy as np
import logging


class WeaviateService(BaseDBConnection, ConfigurableMixin):
    """
    Service for interacting with a Weaviate database.

    Example usage as a Context Manager (recommended):

        async with WeaviateService(host="localhost") as db:
            chunks = await db.get_text_chunks(request)
    """
    # Connection config
    host = ConfigurableValue(
        desc="Weaviate host", user_default="localhost")
    rest_port = ConfigurableValue(
        desc="Weaviate REST port", user_default=8080)
    grpc_port = ConfigurableValue(
        desc="Weaviate gRPC port", user_default=50051)

    # Data model config
    chunks_collection = ConfigurableValue(
        desc="Collection/class name storing text chunks",
        user_default="Chunks_test",
    )
    # Property on chunk objects that links/filters by user collection id
    chunk_user_collection_ref = ConfigurableValue(
        desc="Property on Chunks referencing the user collection",
        user_default="userCollection",
        voluntary=True,
    )
    # Property holding the text in chunk objects
    chunk_text_prop = ConfigurableValue(
        desc="Property name of the text field within the chunks collection",
        user_default="text",
        voluntary=True,
    )

    # Max chunks to retrieve per request
    chunks_limit = ConfigurableValue(
        desc="Max number of chunks to retrieve per request",
        user_default=100000,
        voluntary=True,
    )

    # Hybrid search config
    hybrid_search_alpha = ConfigurableValue(
        desc="Alpha parameter for hybrid search (0.0 = pure keyword search, 1.0 = pure vector search)",
        user_default=0.5,
        voluntary=True,
    )

    # Max vector distance for similarity searches
    max_vector_distance = ConfigurableValue(
        desc="Maximum vector distance for similarity searches",
        user_default=0.5,
        voluntary=True,
    )

    @property
    def client(self) -> WeaviateAsyncClient:
        """Safe accessor for the Weaviate client."""
        if self._client is None:
            raise RuntimeError(
                "WeaviateService is not connected. Use 'with' block "
                "or call connect() method before accessing data."
            )
        return self._client

    def __post_init__(self):
        self._client: WeaviateAsyncClient | None = None
        # await self.connect() // TODO: Discuss if we want auto-connect on init, I prefer explicit connect or context manager usage.

    async def connect(self) -> None:
        """Establishes connection if not already connected."""
        if self._client is None:
            self._client = weaviate.use_async_with_custom(
                http_host=self.host,
                http_port=self.rest_port,
                http_secure=False,
                grpc_host=self.host,
                grpc_port=self.grpc_port,
                grpc_secure=False,
            )
            await self._client.connect()

    async def close(self):
        client = getattr(self, '_client', None)
        if client is not None:
            try:
                await client.close()
            except Exception:
                logging.warning(
                    "Failed to close Weaviate client", exc_info=True)
            finally:
                self._client = None

    async def __aenter__(self) -> "WeaviateService":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def get_text_chunks(self, db_request: DBRequest) -> list[TextChunk]:
        # Access the chunks collection
        chunks_collection = self.client.collections.use(
            self.chunks_collection)

        results: list[TextChunk] = []
        MAX_TOTAL_LIMIT = max(100000, self.chunks_limit)

        # Definition of the filter using reference property
        chunk_filter = Filter.by_ref(self.chunk_user_collection_ref).by_id().equal(db_request.collection_id) if (
            db_request.collection_id is not None
        ) else None

        response = await chunks_collection.query.fetch_objects(
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

    async def find_similar_text_chunks(
        self,
        text: str,
        embedding: np.ndarray,
        db_request: DBRequest | None = None,
        k: int | None = None,
    ) -> list[TextChunk]:
        # Access the chunks collection
        chunks_collection = self.client.collections.use(
            self.chunks_collection)

        results: list[TextChunk] = []
        top_k = k if k is not None else self.chunks_limit

        chunk_filter = Filter.by_ref(
            self.chunk_user_collection_ref).by_id().equal(db_request.collection_id) if (
                db_request is not None and db_request.collection_id is not None
        ) else None

        vec = embedding.tolist()

        response =  await chunks_collection.query.hybrid(
            query=text,
            vector=vec,
            alpha=self.hybrid_search_alpha,
            filters=chunk_filter,
            return_properties=[self.chunk_text_prop],
            limit=top_k,
            max_vector_distance=self.max_vector_distance,
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

    async def get_embeddings(self, text_chunks: list[TextChunk]) -> np.ndarray:
        # If no text chunks, return empty array
        if not text_chunks:
            return np.array([])

        # Access the chunks collection
        chunks_collection = self.client.collections.use(
            self.chunks_collection)

        # We need to extract the UUIDs of the text chunks
        uuids = [chunk.id for chunk in text_chunks]

        response = await chunks_collection.query.fetch_objects_by_ids(
            ids=uuids,
            include_vector=True,
            return_properties=[],
        )

        vector_map = {
            obj.uuid: obj.vector.get("default")
            for obj in response.objects
            if obj.vector
        }

        embeddings = []
        missing_ids = []

        for chunk in text_chunks:
            vector = vector_map.get(chunk.id)
            if vector is not None:
                embeddings.append(vector)
            else:
                missing_ids.append(str(chunk.id))

        if missing_ids:
            raise ValueError(
                f"Embeddings not found for TextChunk IDs: {missing_ids}"
            )

        return np.array(embeddings)
