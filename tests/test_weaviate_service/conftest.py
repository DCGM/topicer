# conftest.py is used to set up fixtures and configurations for the test suite.
import pytest
from unittest.mock import MagicMock
from topicer.database.weaviate_service import WeaviateService
import weaviate.classes.config as wvcc

@pytest.fixture
def mock_service(mocker):
    """
    This 

    :param mocker: The mocker fixture for mocking dependencies
    """

    # Mock the weaviate client connection
    mock_connect = mocker.patch(
        "topicer.database.weaviate_service.weaviate.connect_to_custom")

    # Create a mock client to be returned by the connect function
    mock_client = MagicMock()
    mock_connect.return_value = mock_client

    # Initialize the WeaviateService which will use the mocked client
    service = WeaviateService()

    # Return both the service and the mock client for further configuration in tests
    return service, mock_client


@pytest.fixture
def integration_service():
    # Initialize WeaviateService (it will connect to localhost:8080 by default)
    test_chunk_collection_name = "Test_Chunks_Integration"
    test_user_collection_name = "Test_UserCollection"
    service = WeaviateService(chunks_collection=test_chunk_collection_name)
    client = service._client

    # --- SETUP: Create schema ---
    # First, delete any old collection that might be left over from previous tests
    client.collections.delete(test_chunk_collection_name)
    client.collections.delete(test_user_collection_name)

    # Create the collection we will reference (UserCollection)
    client.collections.create(name=test_user_collection_name)
    
    # Create the main collection with texts and reference
    client.collections.create(
        name=test_chunk_collection_name,
        properties=[
            wvcc.Property(name="text", data_type=wvcc.DataType.TEXT),
        ],
        # Define the reference that your code in 'get_text_chunks' filters
        references=[
            wvcc.ReferenceProperty(
                name=service.chunk_user_collection_ref,
                target_collection=test_user_collection_name
            )
        ]
    )

    yield service  # Here the actual test runs

    # --- TEARDOWN: Clean up after the test ---
    client.collections.delete(test_chunk_collection_name)
    client.collections.delete(test_user_collection_name)
    client.close()