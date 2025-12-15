from topicer.base import BaseDBConnection
from classconfig import ConfigurableMixin, ConfigurableValue
import weaviate
from weaviate import WeaviateClient
from weaviate.classes.query import Filter
from topicer.schemas import TextChunk, DBRequest
from uuid import UUID


class WeaviateClient(BaseDBConnection, ConfigurableMixin):
    # Connection config
    host: str = ConfigurableValue(desc="Weaviate host", user_default="localhost")
    rest_port: int = ConfigurableValue(desc="Weaviate REST port", user_default=8080)
    grpc_port: int = ConfigurableValue(desc="Weaviate gRPC port", user_default=50051)

    # Data model config
    chunks_collection: str = ConfigurableValue(
        desc="Collection/class name storing text chunks",
        user_default="Chunks_test",
    )
    user_collection: str = ConfigurableValue(
        desc="Collection/class name for user collections",
        user_default="Usercollection_test",
    )
    # Property on chunk objects that links/filters by user collection id
    chunk_user_collection_id_prop: str = ConfigurableValue(
        desc="Property on Chunks referencing/identifying the user collection ID (UUID as string)",
        user_default="user_collection_id",
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
        user_default=100,
    )

    def __post_init__(self):
        self._client: WeaviateClient = weaviate.connect_to_custom(
            http_host=self.host,
            http_port=self.rest_port,
            grpc_port=self.grpc_port,
        )

    def get_text_chunks(self, db_request: DBRequest) -> list[TextChunk]:
        """
        Fetch chunks, optionally filtered by DBRequest.collection_id via chunk_user_collection_id_prop.

        Returns list of TextChunk where id is a UUID parsed from Weaviate object's id and text from chunk_text_prop.
        """

        chunks_coll = self.chunks_collection
        props = [self.chunk_text_prop]

        where_filter = None
        if db_request.collection_id is not None and self.chunk_user_collection_id_prop:
            where_filter = Filter.by_property(self.chunk_user_collection_id_prop).equal(
                str(db_request.collection_id)
            )

        response = self._client.collections.use(chunks_coll).query.fetch_objects(
            filters=where_filter,
            return_properties=props,
            limit=self.chunks_limit,
        )

        items = []
        for obj in response.objects:
            text = obj.properties.get(self.chunk_text_prop, "")
            object_id = UUID(obj.id)

            items.append(TextChunk(id=object_id, text=text))

        return items

    def find_similar_text_chunks(self, text_chunks, embedding, db_request = None, k = None):
        ...