from classconfig import ConfigurableMixin, ConfigurableValue
from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio

from topicer.base import BaseLLMService


class OpenAIService(BaseLLMService, ConfigurableMixin):
    reasoning: str = ConfigurableValue(
        desc="Reasoning effort level for OpenAI calls", user_default="medium")
    model: str = ConfigurableValue(
        desc="OpenAI model to use", user_default="gpt-5-mini")

    def __post_init__(self):
        # Jen příprava, zatím nic neotevíráme
        self.client: AsyncOpenAI | None = None

    async def close(self):
        if self.client is not None:
            await self.client.close()
            self.client = None
            
    async def connect(self):
        if self.client is None:
            self.client = AsyncOpenAI()

    async def __aenter__(self):
        # Klient se vytvoří až když je potřeba
        if self.client is None:
            self.client = AsyncOpenAI()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.close()

    async def process_text_chunks(self, text_chunks: list[str], instruction: str, model: str | None = None) -> list[str]:
        raise NotImplementedError(
            "OpenAIService.process_text_chunks is not implemented yet.")

    async def process_text_chunks_structured(self, text_chunks: list[str], instruction: str, output_type: type[BaseModel], model: str | None = None) -> list[BaseModel]:

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

        tasks = [proces_single_chunk(tc) for tc in text_chunks]
        results = await asyncio.gather(*tasks)

        return [res for res in results]
