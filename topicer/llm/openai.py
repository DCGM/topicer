from classconfig import ConfigurableMixin, ConfigurableValue
from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio

from rich.console import Console
from rich.status import Status

from topicer.base import BaseLLMService


class OpenAIService(BaseLLMService, ConfigurableMixin):
    reasoning: str = ConfigurableValue(
        desc="Reasoning effort level for OpenAI calls", user_default="medium")
    model: str = ConfigurableValue(
        desc="OpenAI model to use", user_default="gpt-5-mini")

    def __post_init__(self):
        # Jen příprava, zatím nic neotevíráme
        self._client: AsyncOpenAI | None = None
        
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
        if self._client is not None:
            await self._client.close()
            self._client = None
            
    async def connect(self):
        if self._client is None:
            self._client = AsyncOpenAI()

    async def __aenter__(self):
        if self._client is None:
            self._client = AsyncOpenAI()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client is not None:
            await self.close()

    async def process_text_chunks(self, text_chunks: list[str], instruction: str, model: str | None = None) -> list[str]:
        async def proces_single_chunk(text_chunk: str) -> str:
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
            return response.choices[0].message.content or ""

        tasks = [proces_single_chunk(tc) for tc in text_chunks]
        results = await asyncio.gather(*tasks)

        return [res for res in results]

    async def process_text_chunks_structured(self, text_chunks: list[str], instruction: str, output_type: type[BaseModel], model: str | None = None) -> list[BaseModel]:
        console = Console()

        async def proces_single_chunk(text_chunk: str) -> BaseModel:
            response = await self.client.responses.parse(
                model=model or self.model,
                instructions=instruction,
                input=text_chunk,
                text_format=output_type,
                reasoning={
                    "effort": self.reasoning
                }
            )
            return response.output_parsed

        with console.status("[bold green]Waiting for response from OpenAI LLM model", spinner="dots"):
            tasks = [proces_single_chunk(tc) for tc in text_chunks]
            results = await asyncio.gather(*tasks)

        return [res for res in results]
