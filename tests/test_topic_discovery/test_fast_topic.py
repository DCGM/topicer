import uuid

import numpy as np

from topicer.schemas import TextChunk
from topicer.topic_discovery.fast_topic import FastTopicDiscovery


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
    FastTopicDiscovery.truncate_texts(texts, max_length)

    assert texts[0].text == "Short text."
    assert texts[1].text == "This is a bit longer"
    assert texts[2].text == "Tiny."
