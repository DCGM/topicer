from unittest.mock import MagicMock
from uuid import uuid4
from topicer.schemas import TextChunk
import numpy as np
import pytest

# -------------------- UNIT TESTS --------------------

@pytest.mark.unit
def test_get_embeddings_value_error_if_missing_unit(mock_service):
    service, mock_client = mock_service
    
    # Simulate that Weaviate returns no objects for the given IDs
    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection
    mock_collection.query.fetch_objects_by_ids.return_value.objects = []

    # Tested ID
    missing_id = uuid4()
    chunks = [TextChunk(id=missing_id, text="Chybějící")]

    # Verify that the method raises a ValueError
    with pytest.raises(ValueError) as excinfo:
        service.get_embeddings(chunks)
    
    assert "Embeddings not found" in str(excinfo.value)
    assert str(missing_id) in str(excinfo.value)

@pytest.mark.unit
def test_get_embeddings_empty_input_unit(mock_service):
    service, _ = mock_service
    # Verify that for empty input it returns an empty array and does not call the DB
    result = service.get_embeddings([])
    assert isinstance(result, np.ndarray)
    assert len(result) == 0
    
@pytest.mark.unit
def test_get_embeddings_success_unit(mock_service):
    service, mock_client = mock_service
    
    # Preparation of test data
    id1 = uuid4()
    id2 = uuid4()
    vec1 = [0.1, 0.2, 0.3]
    vec2 = [0.4, 0.5, 0.6]
    
    chunks = [
        TextChunk(id=id1, text="First"),
        TextChunk(id=id2, text="Second")
    ]

    # Simulation of response from Weaviate
    # We need to create objects that have attributes .uuid and .vector["default"]
    mock_obj1 = MagicMock()
    mock_obj1.uuid = id1
    mock_obj1.vector = {"default": vec1}

    mock_obj2 = MagicMock()
    mock_obj2.uuid = id2
    mock_obj2.vector = {"default": vec2}

    mock_collection = MagicMock()
    mock_client.collections.use.return_value = mock_collection
    
    # Set fetch_objects_by_ids to return our mock objects
    mock_collection.query.fetch_objects_by_ids.return_value.objects = [mock_obj1, mock_obj2]

    # 3. Call the method
    result = service.get_embeddings(chunks)

    # 4. Assertions
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)  # 2 chunks, each has 3 numbers in the vector
    
    # Verify that the data in the matrix matches (and is in the correct order)
    np.testing.assert_array_equal(result[0], vec1)
    np.testing.assert_array_equal(result[1], vec2)
    
# -------------------- INTEGRATION TESTS --------------------

def test_get_embeddings_integration(integration_service):
    service = integration_service
    collection = service._client.collections.use(service.chunks_collection)

    # Preparation of test data
    chunk_id = uuid4()
    vector = [0.1, 0.2, 0.3] + [0.0] * 1533  # Simulate 1536-dim vector
    
    # Insert the object directly with the vector into the DB
    collection.data.insert(
        uuid=chunk_id,
        properties={"text": "Test chunk"},
        vector=vector
    )

    # Call the method
    chunks_to_fetch = [TextChunk(id=chunk_id, text="Test chunk")]
    embeddings = service.get_embeddings(chunks_to_fetch)

    # Assertions
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 1536)
    assert np.allclose(embeddings[0], vector, atol=1e-5)
    
@pytest.mark.integration
def test_get_embeddings_multiple_ordering_integration(integration_service):
    """Verify that embeddings are returned in the same order as the input chunks."""
    service = integration_service
    collection = service._client.collections.use(service.chunks_collection)

    # 1. Preparation of 3 chunks with different vectors
    ids = [uuid4() for _ in range(3)]
    vectors = [
        [0.1] * 1536,
        [0.2] * 1536,
        [0.3] * 1536
    ]

    for uid, vec in zip(ids, vectors):
        collection.data.insert(
            uuid=uid,
            properties={"text": f"Chunk {uid}"},
            vector=vec
        )

    # We request the embeddings in a shuffled order
    shuffled_chunks = [
        TextChunk(id=ids[2], text="..."),
        TextChunk(id=ids[0], text="..."),
        TextChunk(id=ids[1], text="...")
    ]
    
    embeddings = service.get_embeddings(shuffled_chunks)

    # 3. Check the order in the matrix
    assert embeddings.shape == (3, 1536)
    np.testing.assert_allclose(embeddings[0], vectors[2], atol=1e-5)
    np.testing.assert_allclose(embeddings[1], vectors[0], atol=1e-5)
    np.testing.assert_allclose(embeddings[2], vectors[1], atol=1e-5)
    
@pytest.mark.integration
def test_get_embeddings_raises_value_error_if_any_missing_integration(integration_service):
    """Verify that if even one ID is missing, the method raises a ValueError with a description."""
    service = integration_service
    collection = service._client.collections.use(service.chunks_collection)

    # Insert one existing
    existing_id = uuid4()
    collection.data.insert(
        uuid=existing_id,
        properties={"text": "Existing chunk"},
        vector=[0.1] * 1536
    )

    # Add one missing ID
    missing_id = uuid4()
    chunks = [
        TextChunk(id=existing_id, text="OK"),
        TextChunk(id=missing_id, text="Missing")
    ]

    # Test raising exception
    with pytest.raises(ValueError) as excinfo:
        service.get_embeddings(chunks)
    
    assert str(missing_id) in str(excinfo.value)
    assert "Embeddings not found" in str(excinfo.value)
    
@pytest.mark.integration
def test_get_embeddings_object_exists_but_no_vector_integration(integration_service):
    """Special case: Object exists in DB but has no vector (default is None)."""
    service = integration_service
    collection = service._client.collections.use(service.chunks_collection)

    chunk_id = uuid4()
    # Insert object WITHOUT vector (if schema allows)
    collection.data.insert(
        uuid=chunk_id,
        properties={"text": "I am here but have no vector"}
        # vector=None (omitted)
    )

    chunks = [TextChunk(id=chunk_id, text="...")]

    with pytest.raises(ValueError) as excinfo:
        service.get_embeddings(chunks)

    assert "Embeddings not found" in str(excinfo.value)
    assert str(chunk_id) in str(excinfo.value)