import pytest
import numpy as np
from unittest.mock import MagicMock
from topicer.schemas import DBRequest, TextChunk
from uuid import uuid4

# -------------------- UNIT TESTS --------------------

@pytest.mark.unit
def test_find_similar_text_chunks_unit(mock_service):
    service, mock_client = mock_service

    # Definition of tue test data
    test_uuid = uuid4()
    test_text = "Some similar text chunk"
    test_embedding = np.array([0.1, 0.2, 0.3])
    db_request = DBRequest(collection_id=uuid4())

    mock_obj = MagicMock()
    mock_obj.uuid = test_uuid
    mock_obj.properties = {service.chunk_text_prop: test_text}

    mock_response = MagicMock()
    mock_response.objects = [mock_obj]

    # Mocking the chain of calls to reach hybrid search
    mock_collection = mock_client.collections.use.return_value
    mock_collection.query.hybrid.return_value = mock_response

    # Action
    results = service.find_similar_text_chunks(
        text="query text",
        embedding=test_embedding,
        db_request=db_request,
        k=5
    )

    # 4. Assertions
    assert len(results) == 1
    assert isinstance(results[0], TextChunk)
    assert results[0].text == test_text
    assert results[0].id == test_uuid

    # Verify that hybrid search was called with correct parameters
    mock_collection.query.hybrid.assert_called_once()
    args, kwargs = mock_collection.query.hybrid.call_args
    assert kwargs['query'] == "query text"
    assert kwargs['limit'] == 5
    assert kwargs['alpha'] == service.hybrid_search_alpha

@pytest.mark.unit
def test_find_similar_text_chunks_no_results_unit(mock_service):
    service, mock_client = mock_service

    test_embedding = np.array([0.5, 0.5, 0.5])

    mock_response = MagicMock()
    mock_response.objects = []

    mock_collection = mock_client.collections.use.return_value
    mock_collection.query.hybrid.return_value = mock_response

    results = service.find_similar_text_chunks(
        text="non-matching query",
        embedding=test_embedding,
        db_request=None,
        k=5
    )

    assert len(results) == 0

# -------------------- INTEGRATION TESTS --------------------

@pytest.mark.integration
def test_find_similar_text_chunks_integration(integration_service):
    service = integration_service
    client = service._client

    # Prepare the data
    user_col_id = uuid4()

    # Create a user collection object (since we filter via references)
    user_col = client.collections.use(service.chunk_user_collection_ref)
    user_col.data.insert(properties={}, uuid=user_col_id)

    # Insert a test chunk with a reference to the user
    chunks_col = client.collections.use(service.chunks_collection)
    chunks_col.data.insert(
        properties={
            service.chunk_text_prop: "This is a sample text about programming."},
        references={service.chunk_user_collection_ref: user_col_id},
        vector=[0.1] * 1536
    )

    # 2. Action
    db_request = DBRequest(collection_id=user_col_id)
    embedding = np.array([0.1] * 1536)

    results = service.find_similar_text_chunks(
        text="programming",
        embedding=embedding,
        db_request=db_request,
        k=10
    )

    # 3. Assertions
    assert len(results) > 0
    assert "programming" in results[0].text

@pytest.mark.integration
def test_find_similar_text_chunks_no_match_integration(integration_service):
    service = integration_service
    client = service._client

    # Prepare the data
    user_col_id = uuid4()

    user_col = client.collections.use(service.chunk_user_collection_ref)
    user_col.data.insert(properties={}, uuid=user_col_id)
    
    # Vector in DB (1.0 at start, rest 0s)
    vector_in_db = [0.0] * 1536
    vector_in_db[0] = 1.0

    # Vector in query (1.0 at index 1, rest 0s)
    query_vector = [0.0] * 1536
    query_vector[1] = 1.0

    chunks_col = client.collections.use(service.chunks_collection)
    chunks_col.data.insert(
        properties={
            service.chunk_text_prop: "This text is about cooking."},
        references={service.chunk_user_collection_ref: user_col_id},
        vector=vector_in_db
    )

    # 2. Action
    db_request = DBRequest(collection_id=user_col_id)
    embedding = np.array(query_vector)

    results = service.find_similar_text_chunks(
        text="quantum physics",
        embedding=embedding,
        db_request=db_request,
        k=10
    )

    # 3. Assertions
    assert len(results) == 0
