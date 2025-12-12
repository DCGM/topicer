import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openai import RateLimitError
from pydantic import BaseModel
from topicer.llm.api_async import APIAsync, OpenAsyncAPI, OllamaAsyncAPI


# --- Helper Artifacts for Testing ---

class TestOutputModel(BaseModel):
    summary: str
    sentiment: int


class MockResponse:
    def __init__(self, output_text=None, output_parsed=None):
        self.output_text = output_text
        self.output_parsed = output_parsed


# --- Fixtures ---


@pytest.fixture
def open_api_instance():
    """Creates an instance of OpenAsyncAPI with mocked client."""
    with patch("topicer.llm.api_async.AsyncOpenAI") as MockClient:
        # Create the instance with dummy config
        instance = OpenAsyncAPI(
            api_key="dummy_key",
            base_url="http://dummy",
            concurrency=2,
            pool_interval=0.1  # Short interval for faster tests
        )
        instance.client = MockClient.return_value
        instance.client.responses = MagicMock()
        instance.client.responses.create = AsyncMock()
        instance.client.responses.parse = AsyncMock()
        return instance


@pytest.fixture
def ollama_api_instance():
    """Creates an instance of OllamaAsyncAPI with mocked client."""
    with patch("topicer.llm.api_async.AsyncClient") as MockClient:
        instance = OllamaAsyncAPI(
            api_key="dummy",
            base_url="http://localhost:11434",
            concurrency=2
        )
        instance.client = MockClient.return_value
        instance.client.chat = AsyncMock()
        return instance


# --- Tests for Static Methods ---

def test_convert_output_to_base_model():
    """Test json_repair capability on malformed JSON."""
    # Malformed JSON (missing quotes around keys, single quotes)
    malformed_json = "{summary: 'Great job', sentiment: 10}"

    result = APIAsync.convert_output_to_base_model(malformed_json, TestOutputModel)

    assert isinstance(result, TestOutputModel)
    assert result.summary == "Great job"
    assert result.sentiment == 10


# --- Tests for OpenAsyncAPI ---

@pytest.mark.asyncio
async def test_openapi_process_single_request_unstructured(open_api_instance):
    """Test standard text generation without pydantic model."""
    mock_response = MockResponse(output_text="Generated Text")
    open_api_instance.client.responses.create.return_value = mock_response

    result = await open_api_instance.process_single_request(
        text_chunk="Input",
        instruction="Do work",
        output_type=None,
        model="test-model"
    )

    assert result == "Generated Text"
    open_api_instance.client.responses.create.assert_called_once()


@pytest.mark.asyncio
async def test_openapi_process_single_request_structured(open_api_instance):
    """Test structured generation returning a Pydantic model."""
    mock_response = MockResponse(output_parsed=TestOutputModel(summary="Structured", sentiment=5))
    open_api_instance.client.responses.parse.return_value = mock_response

    result = await open_api_instance.process_single_request(
        text_chunk="Input",
        instruction="Do structured work",
        output_type=TestOutputModel,
        model="test-model"
    )

    assert isinstance(result, TestOutputModel)
    assert result.summary == "Structured"
    open_api_instance.client.responses.parse.assert_called_once()


@pytest.mark.asyncio
async def test_openapi_rate_limit_retry(open_api_instance):
    """Test that the code sleeps and retries on RateLimitError."""
    mock_response = MockResponse(output_text="Success after retry")

    # Side effect: Raise error first, then return success
    open_api_instance.client.responses.create.side_effect = [
        RateLimitError(message="Rate limit", response=MagicMock(), body=None),
        mock_response
    ]

    # Mock asyncio.sleep to not actually wait during test
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await open_api_instance.process_single_request(
            text_chunk="Input",
            instruction="Instr",
            output_type=None,
            model="test-model"
        )

    assert result == "Success after retry"
    assert open_api_instance.client.responses.create.call_count == 2
    mock_sleep.assert_called_with(open_api_instance.pool_interval)


# --- Tests for OllamaAsyncAPI ---

@pytest.mark.asyncio
async def test_ollama_process_single_request_unstructured(ollama_api_instance):
    """Test Ollama chat completion."""
    # Mocking the specific return structure of ollama client
    mock_return = MagicMock()
    mock_return.message.content = "Ollama says hello"
    ollama_api_instance.client.chat.return_value = mock_return

    result = await ollama_api_instance.process_single_request(
        text_chunk="Input",
        instruction="Chat",
        output_type=None,
        model="ollama-model"
    )

    assert result == "Ollama says hello"
    ollama_api_instance.client.chat.assert_called_once()


@pytest.mark.asyncio
async def test_ollama_process_single_request_structured(ollama_api_instance):
    """Test Ollama structured output."""
    mock_return = MagicMock()
    # Return valid JSON in content
    mock_return.message.content = '{"summary": "Ollama Struct", "sentiment": 1}'
    ollama_api_instance.client.chat.return_value = mock_return

    result = await ollama_api_instance.process_single_request(
        text_chunk="Input",
        instruction="Chat",
        output_type=TestOutputModel,
        model="ollama-model"
    )

    assert isinstance(result, TestOutputModel)
    assert result.summary == "Ollama Struct"

    # specific verification that format was passed as json schema
    call_kwargs = ollama_api_instance.client.chat.call_args.kwargs
    assert "format" in call_kwargs
    assert call_kwargs["format"] == TestOutputModel.model_json_schema()


# --- Integration/Concurrency Tests (Base Class Logic) ---

@pytest.mark.asyncio
async def test_process_requests_concurrency(open_api_instance):
    """
    Test that process_requests yields results correctly using as_completed.
    We reuse open_api_instance but mock process_single_request to control timing/output.
    """

    # We patch process_single_request on the instance to avoid complex API mocks here
    # and focus strictly on the async generator logic
    async def side_effect(chunk, *args, **kwargs):
        return f"Processed {chunk}"

    with patch.object(open_api_instance, 'process_single_request', side_effect=side_effect):
        chunks = ["A", "B", "C"]
        results = []

        async for res in open_api_instance.process_requests(chunks, "inst", None, "test-model"):
            results.append(res)

        # Since it uses as_completed, order isn't guaranteed, but count and content are
        assert len(results) == 3
        assert set(results) == {(0, "Processed A"), (1, "Processed B"), (2, "Processed C")}


@pytest.mark.asyncio
async def test_process_text_chunks_wrappers(open_api_instance):
    """Test the wrapper method process_text_chunks_structured."""

    async def side_effect(*args, **kwargs):
        return "Wrapped Text"

    with patch.object(open_api_instance, 'process_single_request', side_effect=side_effect):
        chunks = ["1", "2"]
        results = await open_api_instance.process_text_chunks(
            chunks, "inst"
        )

        assert len(results) == 2
        assert all(r == "Wrapped Text" for r in results)


@pytest.mark.asyncio
async def test_process_text_chunks_structured_wrappers(open_api_instance):
    """Test the wrapper method process_text_chunks_structured."""

    expected_model = TestOutputModel(summary="Batch", sentiment=9)

    async def side_effect(*args, **kwargs):
        return expected_model

    with patch.object(open_api_instance, 'process_single_request', side_effect=side_effect):
        chunks = ["1", "2"]
        results = await open_api_instance.process_text_chunks_structured(
            chunks, "inst", TestOutputModel
        )

        assert len(results) == 2
        assert all(isinstance(r, TestOutputModel) for r in results)