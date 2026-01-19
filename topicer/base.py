from abc import ABC, abstractmethod
from typing import Sequence, Type

import numpy as np
from classconfig import (
    CreatableMixin,
    ConfigurableMixin,
    ConfigurableSubclassFactory,
    Config,
)
from dotenv import load_dotenv
from pydantic import BaseModel

from topicer.schemas import DBRequest, DiscoveredTopics, DiscoveredTopicsSparse, Tag, TextChunk, \
    TextChunkWithTagSpanProposals


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

    @abstractmethod
    async def discover_topics_sparse(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopicsSparse:
        """
        Discover topics for a collection of texts and return a sparse representation.

        :param texts: Text chunks to propose tags for.
        :param n: Optional number of topics to propose, if None uses the default value.
        :return: DiscoveredTopicsSparse
        """
        ...

    @abstractmethod
    async def discover_topics_dense(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopics:
        """
        Discover topics for a collection of texts and return a dense representation.

        :param texts: Text chunks to propose tags for.
        :param n: Optional number of topics to propose, if None uses the default value.
        :return: DiscoveredTopics
        """
        ...
    
    @abstractmethod
    async def discover_topics_in_db_sparse(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopicsSparse:
        """
        Discover topics based on a database request and return a sparse representation.

        :param db_request: Database request to fetch texts for topic discovery.
        :param n: Optional number of topics to propose, if None uses the default value.
        :return: DiscoveredTopicsSparse
        """
        ...

    @abstractmethod
    async def discover_topics_in_db_dense(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopics:
        """
        Discover topics based on a database request and return a dense representation.

        :param db_request: Database request to fetch texts for topic discovery.
        :param n: Optional number of topics to propose, if None uses the default value.
        :return: DiscoveredTopics
        """
        ...

    @abstractmethod
    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:
        """
        Propose tags for a given text chunk and return the proposals with span indices.
        
        :param text_chunk: A text chunk to propose tags for.
        :param tags: A list of tags to find in text_chunk.
        :return: TextChunkWithTagSpanProposals
        """
        ...

    @abstractmethod
    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTagSpanProposals]:
        """
        Propose tags for texts found in a database based on a database request and return the proposals with span indices.
        
        :param tag: Tag to find in texts.
        :param db_request: Database request to fetch texts for tag proposal.
        :return: list[TextChunkWithTagSpanProposals]
        """
        ...


class BaseLLMService(ABC, ConfigurableMixin):
    """Base interface for LLM services."""
    @abstractmethod
    async def process_text_chunks(self, text_chunks: list[str], instruction: str, model: str | None = None) -> list[str]:
        ...

    @abstractmethod
    async def process_text_chunks_structured(self, text_chunks: list[str], instruction: str, output_type: Type[BaseModel], model: str | None = None) -> list[BaseModel]:
        ...


class BaseDBConnection(ABC, ConfigurableMixin):
    """Base interface for database connections."""

    @abstractmethod
    def get_text_chunks(self, db_request: DBRequest) -> list[TextChunk]:
        """
        Fetch text chunks from the database based on the provided DBRequest.
        
        :param self: Reference to the current instance
        :param db_request: Database request object containing parameters for fetching text chunks
        :type db_request: DBRequest
        :return: Description
        :rtype: list[TextChunk]
        """
        ...
    
    @abstractmethod
    def find_similar_text_chunks(self, text: str, embedding: np.ndarray, db_request: DBRequest | None = None, k: int | None = None) -> list[TextChunk]:
        """
        Find similar text chunks in the database based on the provided text and embedding using hybrid search.
        
        :param self: Reference to the current instance
        :param text: Text to find similar chunks for
        :type text: str
        :param embedding: Embedding vector to use for similarity search
        :type embedding: np.ndarray
        :param db_request: Database request object containing parameters for filtering results
        :type db_request: DBRequest | None
        :param k: Limit on number of similar chunks to return
        :type k: int | None
        :return: List of similar text chunks
        :rtype: list[TextChunk]
        """
        ...

    @abstractmethod
    def get_embeddings(self, text_chunks: list[TextChunk]) -> np.ndarray:
        """
        Retrieve embeddings for the given text chunks from the database.
        
        :param self: Reference to the current instance
        :param text_chunks: List of text chunks to retrieve embeddings for
        :type text_chunks: list[TextChunk]
        :return: Array of embeddings corresponding to the input text chunks
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        ...


class BaseEmbeddingService(ABC, ConfigurableMixin):
    """Base interface for text embedding services."""
    
    @abstractmethod
    def embed(self, text_chunks: list[str] | str, prompt: str | None = None, normalize: bool | None = None) -> np.ndarray:
        ...

    @abstractmethod
    def embed_queries(self, queries: list[str], normalize: bool | None = None) -> np.ndarray:
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
