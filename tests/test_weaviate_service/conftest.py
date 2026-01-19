# conftest.py is used to set up fixtures and configurations for the test suite.
import pytest
from unittest.mock import MagicMock
from topicer.database.weaviate_service import WeaviateService
import weaviate.classes.config as wvcc

@pytest.fixture
def mock_service(mocker):
    # Mock the weaviate client connection
    mock_connect = mocker.patch(
        "topicer.database.weaviate_service.weaviate.connect_to_custom")

    # Create a mock client to be returned by the connect function
    mock_client = MagicMock()
    mock_connect.return_value = mock_client

    # Initialize the WeaviateService which will use the mocked client
    service = WeaviateService()
    service.connect()

    # Return both the service and the mock client for further configuration in tests
    return service, mock_client


@pytest.fixture
def integration_service():
    # Initialize WeaviateService (it will connect to localhost:8080 by default)
    service = WeaviateService()
    service.connect()
    
    client = service._client

    # --- SETUP: Create schema ---
    # First, delete any old collection that might be left over from previous tests
    client.collections.delete(service.chunks_collection)
    client.collections.delete(service.chunk_user_collection_ref)

    # Create the collection we will reference (UserCollection)
    client.collections.create(name=service.chunk_user_collection_ref)
    
    # Create the main collection with texts and reference
    client.collections.create(
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
        service.connect()
    
    service._client.collections.delete(service.chunks_collection)
    service._client.collections.delete(service.chunk_user_collection_ref)
    service.close()