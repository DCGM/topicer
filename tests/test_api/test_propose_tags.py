# -*- coding: UTF-8 -*-
"""
Created on 31.03.26

:author:     Martin Dočekal
"""
import uuid
from urllib.parse import urljoin

import pytest
import requests
import weaviate.classes.config as wvcc

from tests.test_api.conftest import ensure_test_collections, create_test_collection_id
from topicer.schemas import TextChunk, Tag, TextWithSpan, TextChunkWithTagSpanProposals


@pytest.mark.integration
def test_propose_tags(base_url: str) -> None:
    endpoint = urljoin(f"{base_url.rstrip('/')}/", "tags/propose/texts")
    text_chunk = TextChunk(id=uuid.uuid4(), text="This is a sample text about machine learning and AI.")
    payload = {
        "text_chunk": text_chunk.model_dump(mode="json"),
        "tags": [
            Tag(
                id=uuid.uuid4(),
                name="Machine Learning",
                description="A field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
                examples=[TextWithSpan(text="Machine learning is a subset of AI that focuses on building systems that learn from data.", span_start=0, span_end=16)]
            ).model_dump(mode="json"),
            Tag(
                id=uuid.uuid4(),
                name="acronym",
                description="A word formed from the initial letters of a name or by combining initial letters or parts of a series of words.",
                examples=[TextWithSpan(text="AI stands for Artificial Intelligence.", span_start=0, span_end=2) ]
            ).model_dump(mode="json"),
        ]
    }
    response = requests.post(
        endpoint,
        params={'config_name': 'cross_bert_hf', 'n': 2},
        json=payload,
    )

    assert response.status_code == 200, f"{endpoint} is not 200. Response: {response.text}"

    # validate against TextChunkWithTagSpanProposals
    try:
        result = TextChunkWithTagSpanProposals.model_validate_json(response.text)
    except Exception as e:
        pytest.fail(f"Response does not match TextChunkWithTagSpanProposals schema: {e}")

    assert result.id == text_chunk.id
    assert result.text == text_chunk.text, "Returned text does not match input text."
    assert len(result.tag_span_proposals) > 0, "No tag span proposals returned."


@pytest.mark.integration
@pytest.mark.asyncio
async def test_propose_tags_db(base_url: str, db_request_payload: dict) -> None:
    endpoint = urljoin(f"{base_url.rstrip('/')}/", "tags/propose/db")
    tag = Tag(
        id=uuid.uuid4(),
        name="Machine Learning",
        description="A field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
        examples=[
            TextWithSpan(
                text="Machine learning is a subset of AI that focuses on building systems that learn from data.",
                span_start=0,
                span_end=16,
            )
        ],
    )

    payload = {
        "tag": tag.model_dump(mode="json"),
        "db_request": db_request_payload,
    }
    response = requests.post(
        endpoint,
        params={"config_name": "cross_bert_hf"},
        json=payload,
    )

    assert response.status_code == 200, f"{endpoint} is not 200. Response: {response.text}"

    try:
        result = [TextChunkWithTagSpanProposals.model_validate(item) for item in response.json()]
    except Exception as e:
        pytest.fail(f"Response does not match list[TextChunkWithTagSpanProposals] schema: {e}")

    assert len(result) > 0, "No text chunks returned."
    for proposed_chunk in result:
        assert proposed_chunk.tag_span_proposals is not None


