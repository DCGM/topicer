import pytest
from unittest.mock import MagicMock
from topicer.database.weaviate_service import WeaviateService
from topicer.schemas import DBRequest
from topicer.schemas import TextChunk
from uuid import uuid4
import weaviate.classes.config as wvcc

def test_get_text_chunks_success(mock_service):
    # Arrange
    service, mock_client = mock_service

    # Fake collection and query
    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection

    # Fake returned objects
    fake_uuid = uuid4()
    mock_obj = MagicMock()
    mock_obj.uuid = fake_uuid
    mock_obj.properties = {"text": "Ahoj světe"}

    # Responses: first has data, second empty to stop loop
    mock_response = MagicMock()
    mock_response.objects = [mock_obj]
    mock_empty_response = MagicMock()
    mock_empty_response.objects = []

    mock_collection.query.fetch_objects.side_effect = [
        mock_response, mock_empty_response]

    # Act
    request = DBRequest(collection_id=uuid4())
    vysledek = service.get_text_chunks(request)

    # Assert
    assert len(vysledek) == 1
    assert isinstance(vysledek[0], TextChunk)
    assert vysledek[0].text == "Ahoj světe"
    assert vysledek[0].id == fake_uuid
    assert mock_collection.query.fetch_objects.call_count == 1

def test_get_text_chunks_handles_large_volume(mock_service):
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
        mock_obj.properties = {service.chunk_text_prop: f"Text kusu číslo {i}"}
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
    # 1. Verify that all 5000 objects were returned
    assert len(result) == num_objects
    
    # 2. Verify data integrity (spot check first and last)
    assert result[0].text == "Text kusu číslo 0"
    assert result[-1].text == f"Text kusu číslo {num_objects - 1}"
    
    # 3. Verify that all IDs match
    assert {c.id for c in result} == expected_ids

    # 4. KEY CHANGE: fetch_objects is called exactly ONCE
    assert mock_collection.query.fetch_objects.call_count == 1
    
    # 5. Check that we sent a sufficiently high limit to the DB
    args, kwargs = mock_collection.query.fetch_objects.call_args
    assert kwargs['limit'] >= 100000

def test_get_text_chunks_handles_error(mock_service):
    # Arrange
    service, mock_client = mock_service
    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection

    mock_collection.query.fetch_objects.side_effect = RuntimeError("DB error")

    # Act / Assert
    request = DBRequest(collection_id=uuid4())
    with pytest.raises(RuntimeError):
        service.get_text_chunks(request)
    
@pytest.mark.integration
def test_get_text_chunks_real_retrieval(integration_service):
    service = integration_service
    client = service._client
    user_coll = client.collections.use("Test_UserCollection")
    coll = client.collections.use(service.chunks_collection)

    # Vytvoříme objekty v UserCollection nejdřív
    target_user_id = user_coll.data.insert(properties={})
    other_user_id = user_coll.data.insert(properties={})

    # Vložíme 2 objekty, které odpovídají filtru
    coll.data.insert(
        properties={"text": "Tento text chceme"},
        references={service.chunk_user_collection_ref: target_user_id}
    )
    coll.data.insert(
        properties={"text": "Tento taky"},
        references={service.chunk_user_collection_ref: target_user_id}
    )
    # Vložíme 1 objekt, který filtr nesmí najít
    coll.data.insert(
        properties={"text": "Tento nechceme"},
        references={service.chunk_user_collection_ref: other_user_id}
    )

    # Voláme tvou metodu
    request = DBRequest(collection_id=target_user_id)
    results = service.get_text_chunks(request)

    # Ověření
    assert len(results) == 2
    texts = [r.text for r in results]
    assert "Tento text chceme" in texts
    assert "Tento taky" in texts
    assert "Tento nechceme" not in texts