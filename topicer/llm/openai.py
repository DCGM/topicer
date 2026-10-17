from classconfig import ConfigurableMixin, ConfigurableValue
from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio

from rich.console import Console

from topicer.base import BaseLLMService
from topicer.llm.usage import meter_call, openai_usage


class OpenAIService(BaseLLMService, ConfigurableMixin):
    reasoning: str = ConfigurableValue(
        desc="Reasoning effort level for OpenAI calls", user_default="medium")
    model: str = ConfigurableValue(
        desc="OpenAI model to use", user_default="gpt-5-mini")
    prices: dict | None = ConfigurableValue(
        desc="Model prices in USD per million tokens used to estimate the cost of calls when the provider does not "
             "report it: {model: {input_per_million: float, output_per_million: float}}.",
        user_default=None, voluntary=True,
        validator=lambda x: x is None or isinstance(x, dict))

    def __post_init__(self):
        # Jen příprava, zatím nic neotevíráme
        self._client: AsyncOpenAI | None = None
        self._console: Console = Console()
        self._stack_count: int = 0
        
    @property
    def client(self) -> AsyncOpenAI:
        """Safe accessor for the OpenAI client."""
        if self._client is None:
            raise RuntimeError(
                "OpenAIService client is not connected. Use 'with' block "
                "or call connect() method before accessing data."
            )
        return self._client

    async def close(self):
        if self._stack_count > 0:
            self._stack_count -= 1
        if self._stack_count == 0 and self._client is not None:
            await self._client.close()
            self._client = None
            
    async def connect(self):
        if self._client is None:
            self._client = AsyncOpenAI()
        self._stack_count += 1

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def process_text_chunks(self, text_chunks: list[str], instruction: str, model: str | None = None) -> list[str]:
        async def process_single_chunk(text_chunk: str) -> str:
            with meter_call("openai", model or self.model, self.prices) as meter:
                response = await self.client.chat.completions.create(
                    model=model or self.model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": text_chunk}
                    ],

                    # enforce json output
                    response_format={"type": "json_object"},
                    # for results consistency
                    temperature=0
                )
                meter.success(*openai_usage(response))
            return response.choices[0].message.content or ""

        tasks = [process_single_chunk(tc) for tc in text_chunks]
        results = await asyncio.gather(*tasks)

        return [res for res in results]

    async def process_text_chunks_structured(self, text_chunks: list[str], instruction: str, output_type: type[BaseModel], model: str | None = None) -> list[BaseModel]:

        async def process_single_chunk(text_chunk: str) -> BaseModel:
            with meter_call("openai", model or self.model, self.prices) as meter:
                response = await self.client.responses.parse(
                    model=model or self.model,
                    instructions=instruction,
                    input=text_chunk,
                    text_format=output_type,
                    reasoning={
                        "effort": self.reasoning
                    }
                )
                meter.success(*openai_usage(response))
            return response.output_parsed

        with self._console.status("[bold green]Waiting for response from OpenAI LLM model", spinner="dots"):
            tasks = [process_single_chunk(tc) for tc in text_chunks]
            results = await asyncio.gather(*tasks)

        return [res for res in results]
