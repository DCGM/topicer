import os
from abc import ABC, abstractmethod
from pydantic import BaseModel
from classconfig import (
    CreatableMixin,
    ConfigurableMixin,
    ConfigurableSubclassFactory,
    Config,
    ConfigurableValue
)
from dotenv import load_dotenv
import numpy as np

from topicer.schemas import DBRequest


class MissingServiceError(Exception):
    """Raised when a required service is missing."""
    ...


class BaseTopicer(ABC, ConfigurableMixin):
    """Base interface for topicer variants."""

    def set_llm_service(self, llm_service: 'BaseLLMService') -> None:
        self.llm_service = llm_service

    def set_db_connection(self, db_connection: 'BaseDBConnection') -> None:
        self.db_connection = db_connection

    def set_embedding_service(self, embedding_service: 'BaseEmbeddingService') -> None:
        self.embedding_service = embedding_service

    @abstractmethod
    def check_init(self) -> None:
        """Check if all required services are set. Raise MissingServiceError if not."""
        ...


class BaseLLMService(ABC, ConfigurableMixin):
    """Base interface for LLM services."""
    @abstractmethod
    def process_text_chunks(self, text_chunks: list[str], instruction: str, model: str | None, temperature: float | None) -> list[str]:
        ...

    @abstractmethod
    def process_text_chunks_structured(self, text_chunks: list[str], instruction: str, output_type: BaseModel, model: str | None, temperature: float | None) -> list[BaseModel]:
        ...


class BaseDBConnection(ABC, ConfigurableMixin):
    """Base interface for database connections."""

    @abstractmethod
    def get_text_chunks(self, db_request: DBRequest):
        ...
    
    @abstractmethod
    def find_similar_text_chunks(self, text_chunk: str, embedding: np.ndarray):
        ...


class BaseEmbeddingService(ABC, ConfigurableMixin):
    """Base interface for text embedding services."""
    
    @abstractmethod
    def embed_text_chunks(self, text_chunks: list[str] | str) -> np.ndarray:
        ...


class TopicerFactory(CreatableMixin, ConfigurableMixin):
    topicer: BaseTopicer = ConfigurableSubclassFactory(
        BaseTopicer, desc="Variant of topicer to use")
    llm_service: BaseLLMService = ConfigurableSubclassFactory(
        BaseLLMService, desc="LLM service to use", voluntary=True)
    db_connection: BaseDBConnection = ConfigurableSubclassFactory(
        BaseDBConnection, desc="Database connection to use", voluntary=True)
    embedding_service: BaseEmbeddingService = ConfigurableSubclassFactory(
        BaseEmbeddingService, desc="Text embedding service to use", voluntary=True)
    

def factory(cfg: str | dict | Config) -> BaseTopicer:
    load_dotenv()

    f = TopicerFactory.create(cfg)
    topicer = f.topicer
    topicer.set_llm_service(f.llm_service)
    topicer.set_db_connection(f.db_connection)
    topicer.set_embedding_service(f.embedding_service)
    topicer.check_init()

    return topicer
