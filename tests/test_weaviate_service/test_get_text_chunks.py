import pytest
from unittest.mock import MagicMock
from topicer.database.weaviate_service import WeaviateService
from topicer.schemas import DBRequest
from topicer.schemas import TextChunk
from uuid import uuid4
import weaviate.classes.config as wvcc


# -------------------- UNIT TESTS --------------------

@pytest.mark.unit
def test_get_text_chunks_success_unit(mock_service):
    # Arrange
    service, mock_client = mock_service

    # Fake collection and query
    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection

    # Fake returned objects
    fake_uuid = uuid4()
    mock_obj = MagicMock()
    mock_obj.uuid = fake_uuid
    mock_obj.properties = {service.chunk_text_prop: "Hello world"}

    # Responses: first has data, second empty to stop loop
    mock_response = MagicMock()
    mock_response.objects = [mock_obj]

    mock_collection.query.fetch_objects.return_value = mock_response

    # Act
    request = DBRequest(collection_id=uuid4())
    vysledek = service.get_text_chunks(request)

    # Assert
    assert len(vysledek) == 1
    assert isinstance(vysledek[0], TextChunk)
    assert vysledek[0].text == "Hello world"
    assert vysledek[0].id == fake_uuid
    assert mock_collection.query.fetch_objects.call_count == 1
    
@pytest.mark.unit
def test_get_text_chunks_with_filter_unit(mock_service):
    # Arrange
    service, mock_client = mock_service
    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection

    fake_uuid = uuid4()
    mock_obj = MagicMock()
    mock_obj.uuid = fake_uuid
    mock_obj.properties = {service.chunk_text_prop: "Filtered text"}

    mock_response = MagicMock()
    mock_response.objects = [mock_obj]
    mock_collection.query.fetch_objects.return_value = mock_response

    # Act
    collection_id = uuid4()
    request = DBRequest(collection_id=collection_id)
    result = service.get_text_chunks(request)

    # Assert
    assert len(result) == 1
    
    # Verify that the filter was applied correctly
    args, kwargs = mock_collection.query.fetch_objects.call_args
    assert kwargs['filters'] is not None  # Filter should be set

@pytest.mark.unit
def test_get_text_chunks_large_volume_unit(mock_service):
    # --- Arrange ---
    service, mock_client = mock_service
    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection

    # We generate a larger number of fake objects to simulate large data retrieval
    num_objects = 5000
    fake_objects = []
    expected_ids = set()
    
    for i in range(num_objects):
        u_id = uuid4()
        mock_obj = MagicMock()
        mock_obj.uuid = u_id
        mock_obj.properties = {service.chunk_text_prop: f"Text chunk number {i}"}
        fake_objects.append(mock_obj)
        expected_ids.add(u_id)

    # The response now contains all data in a single list
    mock_response = MagicMock()
    mock_response.objects = fake_objects

    # Mock fetch_objects to return all data in a single call
    mock_collection.query.fetch_objects.return_value = mock_response

    # --- Act ---
    request = DBRequest(collection_id=uuid4())
    result = service.get_text_chunks(request)

    # --- Assert ---
    # Verify that all 5000 objects were returned
    assert len(result) == num_objects
    
    # Verify data integrity (spot check first and last)
    assert result[0].text == "Text chunk number 0"
    assert result[-1].text == f"Text chunk number {num_objects - 1}"
    
    # Verify that all IDs match
    assert {c.id for c in result} == expected_ids

    # fetch_objects is called exactly ONCE
    assert mock_collection.query.fetch_objects.call_count == 1
    
    # Verify that we sent a sufficiently high limit to the DB
    args, kwargs = mock_collection.query.fetch_objects.call_args
    assert kwargs['limit'] >= 100000

@pytest.mark.unit
def test_get_text_chunks_handles_error_unit(mock_service):
    # Arrange
    service, mock_client = mock_service
    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection

    mock_collection.query.fetch_objects.side_effect = RuntimeError("DB error")

    # Assert
    request = DBRequest(collection_id=uuid4())
    with pytest.raises(RuntimeError):
        service.get_text_chunks(request)

# -------------------- INTEGRATION TESTS --------------------
    
@pytest.mark.integration
def test_get_text_chunks_success_integration(integration_service):
    service = integration_service
    client = service._client
    user_coll = client.collections.use(service.chunk_user_collection_ref)
    coll = client.collections.use(service.chunks_collection)

    # We create objects in UserCollection first
    target_user_id = user_coll.data.insert(properties={})
    other_user_id = user_coll.data.insert(properties={})

    # We insert 2 objects that match the filter
    coll.data.insert(
        properties={service.chunk_text_prop: "This text we want"},
        references={service.chunk_user_collection_ref: target_user_id}
    )
    coll.data.insert(
        properties={service.chunk_text_prop: "This one too"},
        references={service.chunk_user_collection_ref: target_user_id}
    )
    # We insert 1 object that the filter should NOT find
    coll.data.insert(
        properties={service.chunk_text_prop: "This one should not be found"},
        references={service.chunk_user_collection_ref: other_user_id}
    )

    # We call your method
    request = DBRequest(collection_id=target_user_id)
    results = service.get_text_chunks(request)

    # Assertions
    assert len(results) == 2
    texts = [r.text for r in results]
    assert "This text we want" in texts
    assert "This one too" in texts
    assert "This one should not be found" not in texts
        
@pytest.mark.integration
def test_get_text_chunks_large_volume_integration(integration_service):
    service = integration_service
    client = service._client
    user_coll = client.collections.use(service.chunk_user_collection_ref)
    coll = client.collections.use(service.chunks_collection)

    # Insert a user collection object to reference
    target_user_id = user_coll.data.insert(properties={})

    num_objects = 1500 
    expected_ids = set()

    # Insert a large number of objects
    for i in range(num_objects):
        obj_id = coll.data.insert(
            properties={service.chunk_text_prop: f"Object number {i}"},
            references={service.chunk_user_collection_ref: target_user_id}
        )
        expected_ids.add(obj_id)

    # Action
    request = DBRequest(collection_id=target_user_id)
    results = service.get_text_chunks(request)

    # Assertions
    assert len(results) == num_objects
    assert {c.id for c in results} == expected_ids