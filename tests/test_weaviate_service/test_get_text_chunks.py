import pytest
from unittest.mock import MagicMock
from topicer.database.weaviate_service import WeaviateService
from topicer.schemas import DBRequest
from topicer.schemas import TextChunk
from uuid import uuid4
import weaviate.classes.config as wvcc

@pytest.fixture
def mock_service(mocker):
    """
    This 

    :param mocker: The mocker fixture for mocking dependencies
    """

    mock_connect = mocker.patch(
        "topicer.database.weaviate_service.weaviate.connect_to_custom")

    mock_client = MagicMock()
    mock_connect.return_value = mock_client

    service = WeaviateService()

    return service, mock_client


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
    assert mock_collection.query.fetch_objects.call_count == 2


def test_get_text_chunks_empty_result(mock_service):
    # Arrange
    service, mock_client = mock_service
    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection

    mock_empty_response = MagicMock()
    mock_empty_response.objects = []

    mock_collection.query.fetch_objects.side_effect = [mock_empty_response]

    # Act
    request = DBRequest(collection_id=uuid4())
    vysledek = service.get_text_chunks(request)

    # Assert
    assert vysledek == []
    assert mock_collection.query.fetch_objects.call_count == 1


def test_get_text_chunks_pagination_merges_results(mock_service):
    # Arrange
    service, mock_client = mock_service
    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection

    # Batch 1
    uuid1 = uuid4()
    obj1 = MagicMock()
    obj1.uuid = uuid1
    obj1.properties = {"text": "První"}

    uuid2 = uuid4()
    obj2 = MagicMock()
    obj2.uuid = uuid2
    obj2.properties = {"text": "Druhý"}

    resp1 = MagicMock()
    resp1.objects = [obj1, obj2]

    # Batch 2
    uuid3 = uuid4()
    obj3 = MagicMock()
    obj3.uuid = uuid3
    obj3.properties = {"text": "Třetí"}

    resp2 = MagicMock()
    resp2.objects = [obj3]

    # End
    resp_end = MagicMock()
    resp_end.objects = []

    mock_collection.query.fetch_objects.side_effect = [resp1, resp2, resp_end]

    # Act
    request = DBRequest(collection_id=uuid4())
    vysledek = service.get_text_chunks(request)

    # Assert
    assert len(vysledek) == 3
    assert [c.text for c in vysledek] == ["První", "Druhý", "Třetí"]
    assert {c.id for c in vysledek} == {uuid1, uuid2, uuid3}
    assert mock_collection.query.fetch_objects.call_count == 3

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
        

# 1. FIXTURE: Připraví čisté prostředí pro každý test
@pytest.fixture
def integration_service():
    # Inicializace servisy (připojí se na localhost:8080 dle defaultu)
    # Změníme název kolekce na unikátní testovací název
    test_collection_name = "Test_Chunks_Integration"
    service = WeaviateService(chunks_collection=test_collection_name)
    client = service._client

    # --- SETUP: Vytvoření schématu ---
    # Nejdřív smažeme starou kolekci, kdyby tam zbyla z minula
    client.collections.delete(test_collection_name)
    client.collections.delete("Test_UserCollection")

    # Vytvoříme kolekci, na kterou se budeme odkazovat (UserCollection)
    client.collections.create(name="Test_UserCollection")
    
    # Vytvoříme hlavní kolekci s texty a referencí
    client.collections.create(
        name=test_collection_name,
        properties=[
            wvcc.Property(name="text", data_type=wvcc.DataType.TEXT),
        ],
        # Definujeme referenci, kterou tvůj kód v 'get_text_chunks' filtruje
        references=[
            wvcc.ReferenceProperty(
                name=service.chunk_user_collection_ref,
                target_collection="Test_UserCollection"
            )
        ]
    )

    yield service  # Tady se spustí samotný test

    # --- TEARDOWN: Úklid po testu ---
    client.collections.delete(test_collection_name)
    client.collections.delete("Test_UserCollection")
    client.close()
    
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