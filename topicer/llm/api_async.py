import asyncio
import logging
from abc import abstractmethod
from typing import AsyncGenerator, Optional, Type

import json_repair
from classconfig import ConfigurableValue, CreatableMixin
from classconfig.validators import StringValidator, MinValueIntegerValidator
from ollama import AsyncClient
from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from topicer.base import BaseLLMService
from topicer.llm.usage import meter_call, openai_usage


class APIAsync(BaseLLMService, CreatableMixin):
    """
    Handles asynchronous requests to the API.
    """
    api_key: str = ConfigurableValue(desc="API key.", validator=StringValidator())
    base_url: Optional[str] = ConfigurableValue(desc="Base URL for API.", user_default=None, voluntary=True)
    pool_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds for checking status.",
        user_default=5,
        voluntary=True,
        validator=lambda x: x is None or x > 0)
    process_requests_interval: Optional[int] = ConfigurableValue(
        desc="Interval in seconds between sending requests when processed synchronously.",
        user_default=1,
        voluntary=True,
        validator=lambda x: x is None or x >= 0)
    rate_limit_retries: int = ConfigurableValue(
        desc="Maximum number of retries of a rate-limited request (each waits pool_interval seconds); "
             "the RateLimitError is raised when they are exhausted.",
        user_default=3, voluntary=True, validator=lambda x: isinstance(x, int) and x >= 0
    )
    concurrency: int = ConfigurableValue(
        desc="Maximum number of concurrent requests to the API. This is used with async processing.",
        user_default=10, voluntary=True, validator=MinValueIntegerValidator(1)
    )
    default_model: Optional[str] = ConfigurableValue(
        desc="Default model to use for requests.",
        user_default=None, voluntary=True
    )
    provider: Optional[str] = ConfigurableValue(
        desc="Provider name reported in LLM usage (e.g. openai, openrouter). Defaults to the adapter's provider.",
        user_default=None, voluntary=True
    )
    prices: Optional[dict] = ConfigurableValue(
        desc="Model prices in USD per million tokens used to estimate the cost of calls when the provider does not "
             "report it: {model: {input_per_million: float, output_per_million: float}}.",
        user_default=None, voluntary=True,
        validator=lambda x: x is None or isinstance(x, dict)
    )

    default_provider: str = "unknown"

    @property
    def provider_name(self) -> str:
        return self.provider or self.default_provider

    @abstractmethod
    async def process_single_request(self, text_chunk: str, instruction: str, output_type: Type[BaseModel] | None, model: str) -> str | BaseModel:
        """
        Processes a single request.

        :param text_chunk: Text chunk to process.
        :param instruction: Instruction for processing.
        :param output_type: Expected output type.
        :param model: Model to use for processing.
        :return: Processed request
        """
        ...

    async def process_requests(self, text_chunks: list[str], instruction: str, output_type: Type[BaseModel] | None, model: str) -> AsyncGenerator[tuple[int, str | BaseModel], None]:
        """
        Processes a list of requests.

        :param text_chunks: List of text chunks to process.
        :param instruction: Instruction for processing.
        :param output_type: Expected output type.
        :param model: Model to use for processing.
        :return: An async generator yielding tuples of (index, result), where index is the position of the text chunk and result is the processed output (str or BaseModel).
        """

        async def process_with_index(index: int, chunk: str):
            result = await self.process_single_request(chunk, instruction, output_type, model)
            return index, result

        tasks = []
        for i, chunk in enumerate(text_chunks):
            tasks.append(asyncio.create_task(process_with_index(i, chunk)))

        try:
            for o in asyncio.as_completed(tasks):
                yield await o
        finally:
            # Cancelling the caller does not cancel the tasks behind as_completed; do not leave provider calls running.
            unfinished = [t for t in tasks if not t.done()]
            for t in unfinished:
                t.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)

    async def process_text_chunks(self, text_chunks: list[str], instruction: str, model: str | None = None) -> list[str]:
        if model is None:
            model = self.default_model

        res = [r async for r in self.process_requests(text_chunks, instruction, output_type=None, model=model)]
        return [r[1] for r in sorted(res, key=lambda x: x[0])]

    async def process_text_chunks_structured(self, text_chunks: list[str], instruction: str,
                                             output_type: Type[BaseModel], model: str | None = None) -> list[BaseModel]:
        if model is None:
            model = self.default_model

        res = [r async for r in self.process_requests(text_chunks, instruction, output_type=output_type, model=model)]
        return [r[1] for r in sorted(res, key=lambda x: x[0])]

    @staticmethod
    def convert_output_to_base_model(output: str, output_type: Type[BaseModel]) -> BaseModel:
        """
        Converts the output string to the specified BaseModel type.

        :param output: Output string to convert.
        :param output_type: BaseModel type to convert to.
        :return: Converted BaseModel instance.
        """
        fixed_output = json_repair.loads(output)
        return output_type.model_validate(fixed_output)


class OpenAsyncAPI(APIAsync):
    """
    Handles asynchronous requests to the OpenAI API.
    """
    default_provider = "openai"

    def __post_init__(self):
        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url
        )
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def process_single_request(self, text_chunk: str, instruction: str, output_type: Type[BaseModel] | None, model: str) -> str | BaseModel:
        async with self.semaphore:
            retries = 0
            while True:
                try:
                    with meter_call(self.provider_name, model, self.prices) as meter:
                        if output_type is None:
                            response = await self.client.responses.create(
                                model=model,
                                instructions=instruction,
                                input=text_chunk
                            )
                        else:
                            response = await self.client.responses.parse(
                                model=model,
                                instructions=instruction,
                                input=text_chunk,
                                text_format=output_type
                            )
                        meter.success(*openai_usage(response))
                    return response.output_text if output_type is None else response.output_parsed
                except RateLimitError:
                    # Each attempt is metered; the retry budget bounds the duration and number of attempts.
                    if retries >= self.rate_limit_retries:
                        logging.error(f"Rate limit reached, giving up after {retries} retries.")
                        raise
                    retries += 1
                    logging.error(f"Rate limit reached. Retry {retries}/{self.rate_limit_retries} in {self.pool_interval} seconds.")
                    await asyncio.sleep(self.pool_interval)


class OllamaAsyncAPI(APIAsync):
    """
    Handles asynchronous requests to the Ollama API.
    """
    default_provider = "ollama"

    def __post_init__(self):
        self.client = AsyncClient(host=self.base_url)
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def process_single_request(self, text_chunk: str, instruction: str, output_type: Type[BaseModel] | None, model: str) -> str | BaseModel:
        async with self.semaphore:
            with meter_call(self.provider_name, model, self.prices) as meter:
                response = await self.client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": text_chunk}
                    ],
                    format=output_type if output_type is None else output_type.model_json_schema()
                )
                meter.success(getattr(response, "prompt_eval_count", None), getattr(response, "eval_count", None))

            res = response.message.content
            if output_type is not None:
                res = self.convert_output_to_base_model(res, output_type)
            return res
