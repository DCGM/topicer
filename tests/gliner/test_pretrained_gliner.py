from pathlib import Path
import uuid

from classconfig import Config
import pytest

from topicer.base import factory, BaseTopicer
from topicer.tagging.gliner import GlinerTopicer
from topicer.schemas import TextChunk, Tag, TextChunkWithTagSpanProposals


@pytest.fixture
def config_path():
    return Path(__file__).parent / "config.yaml"


@pytest.fixture
def topicer(config_path):
    return factory(config_path)


@pytest.fixture
def sample_inputs():
    text = TextChunk(
        id = uuid.uuid4(),
        text = "Barack Obama was the 44th president of the United States.",
    )

    tags = [
        Tag(id=uuid.uuid4(), name="PERSON"),
        Tag(id=uuid.uuid4(), name="LOCATION"),
        Tag(id=uuid.uuid4(), name="ORGANIZATION"),
    ]

    return text, tags


def test_init_not_empty(topicer):
    assert topicer is not None


def test_init_correct_cls(topicer):
    assert isinstance(topicer, GlinerTopicer)


def test_init_value_model(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.model == config.untransformed["topicer"]["config"]["model"]


def test_init_value_threshold(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.threshold == config.untransformed["topicer"]["config"]["threshold"]


def test_init_value_multi_label(topicer, config_path):
    config = Config(BaseTopicer).load(config_path)
    assert topicer.multi_label is config.untransformed["topicer"]["config"]["multi_label"]


@pytest.mark.asyncio
async def test_output_type(topicer, sample_inputs):
    result = await topicer.propose_tags(*sample_inputs)
    assert isinstance(result, TextChunkWithTagSpanProposals), f"Expected result to be of type TextChunkWithTagSpanProposals, but got {type(result)}"


@pytest.mark.asyncio
async def test_output_not_empty(topicer, sample_inputs):
    result = await topicer.propose_tags(*sample_inputs)
    assert len(result.tag_span_proposals) > 0, "Expected at least one tag span proposal, but got none."
