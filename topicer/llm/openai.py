from classconfig import ConfigurableMixin, ConfigurableValue
from openai import AsyncOpenAI
from typing import Type
from pydantic import BaseModel


from topicer.base import BaseLLMService


class OpenAIService(BaseLLMService, ConfigurableMixin):
    reasoning: str = ConfigurableValue(
        desc="Reasoning effort level for OpenAI calls", user_default="medium")
    model: str = ConfigurableValue(
        desc="OpenAI model to use", user_default="gpt-5-mini")

    def __post_init__(self):
        self.client: AsyncOpenAI = AsyncOpenAI()
        
    async def process_text_chunks(self, text_chunks: list[str], instruction: str, model: str | None = None) -> list[str]:
        raise NotImplementedError("OpenAIService.process_text_chunks is not implemented yet.")
    
    async def process_text_chunks_structured(self, text_chunks: list[str], instruction: str, output_type: Type[BaseModel], model: str | None = None) -> list[BaseModel]:
        raise NotImplementedError("OpenAIService.process_text_chunks_structured is not implemented yet.")