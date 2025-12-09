from classconfig import ConfigurableMixin, ConfigurableValue
from openai import AsyncOpenAI

from topicer.base import BaseLLMService


class OpenAIService(BaseLLMService, ConfigurableMixin):
    reasoning: str = ConfigurableValue(
        desc="Reasoning effort level for OpenAI calls", user_default="medium")
    model: str = ConfigurableValue(
        desc="OpenAI model to use", user_default="gpt-5-mini")

    def __post_init__(self):
        self.client: AsyncOpenAI = AsyncOpenAI()
