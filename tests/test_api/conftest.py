import logging

import pytest

from topicer import WeaviateService
from weaviate.classes.query import Filter
import weaviate.classes.config as wvcc


logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--api-url",
        action="store",
        default="http://127.0.0.1:8000/v1",
        help="Base URL for the API to be tested.",
    )
    parser.addoption(
        "--db-host",
        action="store",
        default="localhost",
        help="Database host for integration tests.",
    )

    parser.addoption(
        "--db-port",
        action="store",
        default="8080",
        help="Database port for integration tests.",
    )

    parser.addoption(
        "--db-grpc-port",
        action="store",
        default="50051",
        help="Database gRPC port for integration tests.",
    )


# Create a fixture that returns the base URL based on the environment
@pytest.fixture(scope="session")
def base_url(request):
    api_url = request.config.getoption("--api-url")

    return api_url


@pytest.fixture(scope="function")
async def database_connection(request):
    # Setup code for database connection
    service = WeaviateService(
        host=request.config.getoption("--db-host"),
        rest_port=int(request.config.getoption("--db-port")),
        grpc_port=int(request.config.getoption("--db-grpc-port")),
    )
    async with service:
        yield service


TEST_COLLECTION_PROPERTIES = {
    "name": "test",
    "user_id": "test",
    "description": "collection for automatic tests",
    "color": "#1976d2",
}

TEST_CHUNK_TEXTS = [
    "This is a sample text about machine learning and AI.",
    "Another text discussing the advancements in natural language processing.",
    "A brief overview of deep learning techniques and their applications.",
    "Dog is a great pet and companion for humans.",
    "Cats are independent animals and often kept as indoor pets.",
    "Birds can fly and are known for their beautiful songs.",
]

async def ensure_test_collections(database_connection) -> None:
    user_collection = database_connection.client.collections.use(database_connection.chunk_user_collection_ref)
    if not await user_collection.exists():
        await database_connection.client.collections.create(name=database_connection.chunk_user_collection_ref)

    chunks_collection = database_connection.client.collections.use(database_connection.chunks_collection)
    if not await chunks_collection.exists():
        await database_connection.client.collections.create(
            name=database_connection.chunks_collection,
            properties=[
                wvcc.Property(name=database_connection.chunk_text_prop, data_type=wvcc.DataType.TEXT),
            ],
            references=[
                wvcc.ReferenceProperty(
                    name=database_connection.chunk_user_collection_ref,
                    target_collection=database_connection.chunk_user_collection_ref,
                )
            ],
        )


async def create_test_collection_id(database_connection) -> str:
    """
    Create a test collection and return its ID.
    Fails if a test collection with the same properties already exists (for safety).
    
    Raises:
        RuntimeError: If a test collection with the same properties already exists
    """
    user_collection = database_connection.client.collections.use(database_connection.chunk_user_collection_ref)

    response = await user_collection.query.fetch_objects(
        return_properties=list(TEST_COLLECTION_PROPERTIES.keys()),
        filters=(
            Filter.by_property("user_id").equal(TEST_COLLECTION_PROPERTIES["user_id"]) &
            Filter.by_property("name").equal(TEST_COLLECTION_PROPERTIES["name"])
        )
    )

    # Fail if test collection already exists
    for obj in response.objects:
        properties = obj.properties or {}
        if all(properties.get(key) == value for key, value in TEST_COLLECTION_PROPERTIES.items()):
            collection_uuid = str(obj.uuid)
            print(f"\n⚠️  Test collection already exists: {collection_uuid}")
            print(f"    Properties: {properties}\n")

            user_choice = input("Do you want to delete it and continue? (yes/no): ").strip().lower()

            if user_choice in ("yes", "y"):
                logger.info(f"User chose to delete existing collection: {collection_uuid}")
                await clean_test_database(database_connection, collection_uuid)
                break
            else:
                raise RuntimeError(
                    f"Test collection with properties {TEST_COLLECTION_PROPERTIES} already exists. "
                    f"Please clean up the database before running tests. Collection ID: {collection_uuid}"
                )

    logger.info(f"Test collection not found. Creating a new one for user: {TEST_COLLECTION_PROPERTIES.get('user_id')}")

    collection_id = await user_collection.data.insert(properties=TEST_COLLECTION_PROPERTIES)
    await _populate_collection_with_test_chunks_if_empty(database_connection, collection_id)

    return str(collection_id)


async def _populate_collection_with_test_chunks_if_empty(database_connection, collection_id) -> None:
    chunks_collection = database_connection.client.collections.use(database_connection.chunks_collection)
    existing_chunks = await chunks_collection.query.fetch_objects(
        limit=1,
        filters=Filter.by_ref(database_connection.chunk_user_collection_ref).by_id().equal(collection_id),
    )

    if existing_chunks.objects:
        return

    logger.info(f"Populating collection {collection_id} with {len(existing_chunks.objects)} chunks")
    for text in TEST_CHUNK_TEXTS:
        await chunks_collection.data.insert(
            properties={database_connection.chunk_text_prop: text},
            references={database_connection.chunk_user_collection_ref: collection_id},
        )


async def clean_test_database(database_connection, collection_id: str) -> None:
    """Clean up test data from the database after tests."""
    try:
        # Delete all chunks associated with the test collection
        chunks_collection = database_connection.client.collections.use(database_connection.chunks_collection)
        await chunks_collection.data.delete_many(
            where=Filter.by_ref(database_connection.chunk_user_collection_ref).by_id().equal(collection_id),
        )
        
        # Delete the test collection itself
        user_collection = database_connection.client.collections.use(database_connection.chunk_user_collection_ref)
        await user_collection.data.delete_by_id(collection_id)

        logger.info(f"Cleaned up test data for collection {collection_id}")
    except Exception as e:
        logger.error(f"Error cleaning up test database: {e}")


@pytest.fixture
async def db_request_payload(database_connection):
    await ensure_test_collections(database_connection)
    collection_id = await create_test_collection_id(database_connection)

    # Yield the payload for the test to use
    yield {"collection_id": collection_id}

    # Cleanup after test completes
    await clean_test_database(database_connection, collection_id)


