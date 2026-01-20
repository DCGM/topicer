from unittest.mock import MagicMock

import pytest
import numpy as np

import uuid

from topicer.schemas import TextChunk, DiscoveredTopicsSparse, Topic
from topicer.topic_discovery.fast_topic import FastTopicDiscovery
from topicer.schemas import DBRequest


def test_get_top_k_docs_per_topic():
    text_chunks = [
        TextChunk(
            id=uuid.uuid4(),
            text=f"Document {i}"
        ) for i in range(3)
    ]
    topic_doc_dist = np.array([
        [0, 1, 2],
        [2, 1, 0],
        [0, 2, 1]
    ])

    top_k = FastTopicDiscovery.get_top_k_docs_per_topic(topic_doc_dist, text_chunks, k=2)

    assert len(top_k) == 3  # 3 topics

    for target_docs, selected_docs in zip([["Document 2", "Document 1"], ["Document 0", "Document 1"], ["Document 2", "Document 1"]], top_k):
        assert len(selected_docs) == 2  # top 2 documents per topic
        assert set(target_docs) == set(d.text for d in selected_docs)


def test_get_top_k_docs_per_topic_larger_k():
    text_chunks = [
        TextChunk(
            id=uuid.uuid4(),
            text=f"Document {i}"
        ) for i in range(2)
    ]
    topic_doc_dist = np.array([
        [0, 1],
        [1, 0]
    ])

    top_k = FastTopicDiscovery.get_top_k_docs_per_topic(topic_doc_dist, text_chunks, k=5)

    assert len(top_k) == 2  # 2 topics

    for selected_docs in top_k:
        assert len(selected_docs) == 2  # only 2 documents available
        assert {"Document 0", "Document 1"} == set(d.text for d in selected_docs)


def test_sparsify_topic_document_distribution_threshold_only():
    topic_doc_dist = np.array([
        [0.1, 0.5, 0.4],
        [0.3, 0.2, 0.5]
    ])

    sparse_dist = FastTopicDiscovery.sparsify_topic_document_distribution(
        topic_doc_dist,
        threshold=0.3,
        k=None
    )

    expected = [
        [(1, 0.5), (2, 0.4)],
        [(0, 0.3), (2, 0.5)]
    ]

    assert sparse_dist == expected


def test_sparsify_topic_document_distribution_k_only():
    topic_doc_dist = np.array([
        [0.1, 0.5, 0.4],
        [0.3, 0.2, 0.5]
    ])

    sparse_dist = FastTopicDiscovery.sparsify_topic_document_distribution(
        topic_doc_dist,
        threshold=0.0,
        k=2
    )

    expected = [
        [(1, 0.5), (2, 0.4)],
        [(0, 0.3), (2, 0.5)]
    ]

    assert sparse_dist == expected


def test_sparsify_topic_document_distribution_threshold_and_k():
    topic_doc_dist = np.array([
        [0.1, 0.5, 0.2],
        [0.3, 0.2, 0.5]
    ])

    sparse_dist = FastTopicDiscovery.sparsify_topic_document_distribution(
        topic_doc_dist,
        threshold=0.25,
        k=2
    )

    expected = [
        [(1, 0.5)],
        [(0, 0.3), (2, 0.5)]
    ]

    assert sparse_dist == expected


def test_truncate_texts():
    texts = [
        TextChunk(id=uuid.uuid4(), text="Short text."),
        TextChunk(id=uuid.uuid4(), text="This is a bit longer text that should be truncated."),
        TextChunk(id=uuid.uuid4(), text="Tiny."),
    ]

    max_length = 20
    texts = FastTopicDiscovery.truncate_texts(texts, max_length)

    assert texts[0].text == "Short text."
    assert texts[1].text == "This is a bit longer"
    assert texts[2].text == "Tiny."


@pytest.fixture
def mock_fast_topic_discovery(mocker):
    mock_db_connection = mocker.MagicMock()

    # Mock the methods used in FastTopicDiscovery
    mock_db_connection.get_text_chunks.return_value = [
        TextChunk(id=uuid.uuid4(), text="Sample text 1"),
        TextChunk(id=uuid.uuid4(), text="Sample text 2"),
        TextChunk(id=uuid.uuid4(), text="Sample text 3"),
    ]

    mock_db_connection.get_embeddings.return_value = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])

    fast_topic_discovery = FastTopicDiscovery()
    fast_topic_discovery.set_db_connection(mock_db_connection)
    fast_topic_discovery._get_topics = mocker.AsyncMock(
        return_value=(["Topic 1", "Topic 2"], np.array([[0.6, 0.4, 0.2], [0.2, 0.4, 0.6]]))
    )
    fast_topic_discovery._process_topics = mocker.AsyncMock(
        return_value=DiscoveredTopicsSparse(
            topics=[
                Topic(
                    name="Topic 1",
                    name_explanation="Explanation 1",
                    description="Description 1",
                ),
                Topic(
                    name="Topic 2",
                    name_explanation="Explanation 2",
                    description="Description 2",
                ),
            ],
            topic_documents=[
                [(1, 0.4), (0, 0.6)],
                [(2, 0.6), (1, 0.4)],
            ],
        )
    )

    return fast_topic_discovery


@pytest.mark.asyncio
async def test_discover_topics_in_db_sparse(
    mock_fast_topic_discovery: FastTopicDiscovery,
):
    db_request = DBRequest(
        collection_id=uuid.UUID("123e4567-e89b-12d3-a456-426614174000")
    )

    _ = await mock_fast_topic_discovery.discover_topics_in_db_sparse(
        db_request=db_request,
        db_embeddings=True,
    )

    mock_fast_topic_discovery.db_connection: MagicMock
    mock_fast_topic_discovery.db_connection.get_text_chunks.assert_called_once_with(
        db_request,
    )

    mock_fast_topic_discovery.db_connection.get_embeddings.assert_called_once_with(
        mock_fast_topic_discovery.db_connection.get_text_chunks.return_value,
    )






