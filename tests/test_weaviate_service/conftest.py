# conftest.py is used to set up fixtures and configurations for the test suite.
import pytest
from unittest.mock import AsyncMock
from topicer.database.weaviate_service import WeaviateService
import weaviate.classes.config as wvcc

@pytest.fixture
async def mock_service(mocker):
    # Mock the weaviate client connection
    mock_connect = mocker.patch(
        "topicer.database.weaviate_service.weaviate.use_async_with_custom")

    # Create a mock client to be returned by the connect function
    mock_client = AsyncMock()
    mock_connect.return_value = mock_client

    # Initialize the WeaviateService which will use the mocked client
    service = WeaviateService()
    await service.connect()
    
    # Return both the service and the mock client for further configuration in tests
    yield service, mock_client

    await service.close()

@pytest.fixture
async def integration_service():
    # Initialize WeaviateService (it will connect to localhost:8080 by default)
    service = WeaviateService()
    await service.connect()
    
    client = service._client

    # --- SETUP: Create schema ---
    # First, delete any old collection that might be left over from previous tests
    await client.collections.delete(service.chunks_collection)
    await client.collections.delete(service.chunk_user_collection_ref)

    # Create the collection we will reference (UserCollection)
    await client.collections.create(name=service.chunk_user_collection_ref)
    
    # Create the main collection with texts and reference
    await client.collections.create(
        name=service.chunks_collection,
        properties=[
            wvcc.Property(name="text", data_type=wvcc.DataType.TEXT),
        ],
        # Define the reference that your code in 'get_text_chunks' filters
        references=[
            wvcc.ReferenceProperty(
                name=service.chunk_user_collection_ref,
                target_collection=service.chunk_user_collection_ref
            )
        ]
    )

    yield service  # Here the actual test runs

    # --- TEARDOWN: Recreate connection if closed in test ---
    if service._client is None:
        await service.connect()
    
    await service._client.collections.delete(service.chunks_collection)
    await service._client.collections.delete(service.chunk_user_collection_ref)
    await service.close()