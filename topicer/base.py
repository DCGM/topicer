import os
from abc import ABC
from classconfig import (
    CreatableMixin,
    ConfigurableMixin,
    ConfigurableSubclassFactory,
    Config,
    ConfigurableValue
)
from dotenv import load_dotenv
from openai import AsyncOpenAI


class BaseTopicer(ABC, ConfigurableMixin):
    """Base interface for topicer variants."""

    def set_external_client(self, external_service: 'BaseExternalService'):
        self.external_service = external_service


class BaseExternalService(ABC, ConfigurableMixin):
    ...


class OpenAIService(BaseExternalService, ConfigurableMixin):
    reasoning: str = ConfigurableValue(
        desc="Reasoning effort level for OpenAI calls", user_default="medium")
    model: str = ConfigurableValue(
        desc="OpenAI model to use", user_default="gpt-5-mini")

    def __post_init__(self):
        self.client: AsyncOpenAI = AsyncOpenAI()


class TopicerFactory(CreatableMixin, ConfigurableMixin):
    topicer: BaseTopicer = ConfigurableSubclassFactory(
        BaseTopicer, desc="Variant of topicer to use")
    external_service: BaseExternalService = ConfigurableSubclassFactory(
        BaseExternalService, desc="External service client to use", voluntary=True)


def factory(cfg: str | dict | Config) -> BaseTopicer:
    load_dotenv()

    f = TopicerFactory.create(cfg)
    topicer = f.topicer
    topicer.set_external_client(f.external_service)
    return topicer
