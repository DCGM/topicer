import uuid

import pytest
import requests
import weaviate.classes.config as wvcc
from urllib.parse import urljoin

from topicer.schemas import TextChunk


def _assert_common_topic_response(result: dict, expected_topics: int = 2) -> None:
    assert "topics" in result
    assert "topic_documents" in result
    assert len(result["topics"]) == expected_topics
    assert all(
        "name" in topic and "description" in topic and "name_explanation" in topic
        for topic in result["topics"]
    )


async def _ensure_test_collections(database_connection) -> None:
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

@pytest.fixture
def text_chunks_payload() -> list[dict]:
    chunks = [
        TextChunk(id=uuid.uuid4(), text="This is a sample text about machine learning and AI."),
        TextChunk(id=uuid.uuid4(), text="Another text discussing the advancements in natural language processing."),
        TextChunk(id=uuid.uuid4(), text="A brief overview of deep learning techniques and their applications."),
        TextChunk(id=uuid.uuid4(), text="Dog is a great pet and companion for humans."),
        TextChunk(id=uuid.uuid4(), text="Cats are independent animals and often kept as indoor pets."),
        TextChunk(id=uuid.uuid4(), text="Birds can fly and are known for their beautiful songs."),
    ]
    return [chunk.model_dump(mode="json") for chunk in chunks]


@pytest.fixture
async def db_request_payload(database_connection, text_chunks_payload: list[dict]) -> dict:
    await _ensure_test_collections(database_connection)

    collection_id = "dcedd3ad-fc99-4c85-a968-97df777c2064"
    user_collection = database_connection.client.collections.use(database_connection.chunk_user_collection_ref)

    exists = await user_collection.data.exists(collection_id)

    assert exists, (
        f"Test collection with ID {collection_id} does not exist in the database. "
        "Please ensure it is created and contains the necessary data for testing."
    )

    return {"collection_id": str(collection_id)}


@pytest.mark.integration
def test_discover_texts_sparse(base_url: str, text_chunks_payload: list[dict]) -> None:
    endpoint = urljoin(f"{base_url.rstrip('/')}/", "topics/discover/texts/sparse")
    response = requests.post(
        endpoint,
        params={'config_name': 'fast_topic', 'n': 2},
        json=text_chunks_payload,
    )

    assert response.status_code == 200, f"{endpoint} is not 200. Response: {response.text}"

    result = response.json()
    _assert_common_topic_response(result)

    # it should be sparse representation of a matrix list[list[tuple[int, float]]] N x K matrix where N is number of topics and K is number of documents, the tuple is (document_index, probability)

    assert isinstance(result["topic_documents"], list), (
        f"`topic_documents` must be a list, got {type(result['topic_documents']).__name__}"
    )
    assert len(result["topic_documents"]) == 2, (
        f"Expected 2 topic entries, got {len(result['topic_documents'])}"
    )

    for topic_idx, topic_doc in enumerate(result["topic_documents"]):
        assert isinstance(topic_doc, list), (
            f"`topic_documents[{topic_idx}]` must be a list, "
            f"got {type(topic_doc).__name__}"
        )

        for pair_idx, pair in enumerate(topic_doc):
            assert isinstance(pair, list), (
                f"`topic_documents[{topic_idx}][{pair_idx}]` must be a list, "
                f"got {type(pair).__name__}"
            )

            assert len(pair) == 2, (
                f"`topic_documents[{topic_idx}][{pair_idx}]` must have length 2, "
                f"got {len(pair)}: {pair!r}"
            )
            assert isinstance(pair[0], int), (
                f"Index at `topic_documents[{topic_idx}][{pair_idx}][0]` "
                f"must be int, got {type(pair[0]).__name__}: {pair!r}"
            )
            assert isinstance(pair[1], float), (
                f"Probability at `topic_documents[{topic_idx}][{pair_idx}][1]` "
                f"must be float, got {type(pair[1]).__name__}: {pair!r}"
            )


@pytest.mark.integration
def test_discover_texts_dense(base_url: str, text_chunks_payload: list[dict]) -> None:

    endpoint = urljoin(f"{base_url.rstrip('/')}/", "topics/discover/texts/dense")
    response = requests.post(
        endpoint,
        params={"config_name": "fast_topic", "n": 2},
        json=text_chunks_payload,
    )

    assert response.status_code == 200, f"{endpoint} is not 200. Response: {response.text}"

    result = response.json()
    _assert_common_topic_response(result)

    # Dense representation: list[list[float]] with shape N x K
    # N = number of topics, K = number of documents.
    assert isinstance(result["topic_documents"], list), (
        f"`topic_documents` must be a list, got {type(result['topic_documents']).__name__}"
    )
    assert len(result["topic_documents"]) == 2, (
        f"Expected 2 topic entries, got {len(result['topic_documents'])}"
    )

    for topic_idx, topic_doc in enumerate(result["topic_documents"]):
        assert isinstance(topic_doc, list), (
            f"`topic_documents[{topic_idx}]` must be a list, "
            f"got {type(topic_doc).__name__}"
        )
        assert len(topic_doc) == len(text_chunks_payload), (
            f"`topic_documents[{topic_idx}]` must have {len(text_chunks_payload)} probabilities, "
            f"got {len(topic_doc)}"
        )

        for doc_idx, prob in enumerate(topic_doc):
            assert isinstance(prob, (int, float)), (
                f"`topic_documents[{topic_idx}][{doc_idx}]` must be numeric, "
                f"got {type(prob).__name__}: {prob!r}"
            )
            assert 0.0 <= float(prob) <= 1.0, (
                f"`topic_documents[{topic_idx}][{doc_idx}]` out of [0, 1] range: {prob!r}"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_discover_db_sparse(base_url: str, db_request_payload: dict) -> None:
    endpoint = urljoin(f"{base_url.rstrip('/')}/", "topics/discover/db/sparse")
    response = requests.post(
        endpoint,
        params={"config_name": "fast_topic", "n": 2},
        json=db_request_payload,
    )

    assert response.status_code == 200, f"{endpoint} is not 200. Response: {response.text}"

    result = response.json()
    _assert_common_topic_response(result)

    assert isinstance(result["topic_documents"], list), (
        f"`topic_documents` must be a list, got {type(result['topic_documents']).__name__}"
    )
    assert len(result["topic_documents"]) == 2, (
        f"Expected 2 topic entries, got {len(result['topic_documents'])}"
    )

    for topic_idx, topic_doc in enumerate(result["topic_documents"]):
        assert isinstance(topic_doc, list), (
            f"`topic_documents[{topic_idx}]` must be a list, "
            f"got {type(topic_doc).__name__}"
        )

        for pair_idx, pair in enumerate(topic_doc):
            assert isinstance(pair, list), (
                f"`topic_documents[{topic_idx}][{pair_idx}]` must be a list, "
                f"got {type(pair).__name__}"
            )
            assert len(pair) == 2, (
                f"`topic_documents[{topic_idx}][{pair_idx}]` must have length 2, "
                f"got {len(pair)}: {pair!r}"
            )
            assert isinstance(pair[0], int), (
                f"Index at `topic_documents[{topic_idx}][{pair_idx}][0]` "
                f"must be int, got {type(pair[0]).__name__}: {pair!r}"
            )
            assert isinstance(pair[1], float), (
                f"Probability at `topic_documents[{topic_idx}][{pair_idx}][1]` "
                f"must be float, got {type(pair[1]).__name__}: {pair!r}"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_discover_db_dense(base_url: str, db_request_payload: dict) -> None:
    endpoint = urljoin(f"{base_url.rstrip('/')}/", "topics/discover/db/dense")
    response = requests.post(
        endpoint,
        params={"config_name": "fast_topic", "n": 2},
        json=db_request_payload,
    )

    assert response.status_code == 200, f"{endpoint} is not 200. Response: {response.text}"

    result = response.json()
    _assert_common_topic_response(result)

    assert isinstance(result["topic_documents"], list), (
        f"`topic_documents` must be a list, got {type(result['topic_documents']).__name__}"
    )
    assert len(result["topic_documents"]) == 2, (
        f"Expected 2 topic entries, got {len(result['topic_documents'])}"
    )

