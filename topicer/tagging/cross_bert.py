from typing import Sequence
from pathlib import Path

import torch
from classconfig import ConfigurableMixin, ConfigurableValue

from topicer.base import BaseTopicer, MissingServiceError
from topicer.schemas import DiscoveredTopicsSparse, TagSpanProposal, TextChunk, DiscoveredTopics, DBRequest, Tag, TextChunkWithTagSpanProposals


class CrossBertTopicer(BaseTopicer, ConfigurableMixin):
    model: Path = ConfigurableValue(desc="")
    threshold: float = ConfigurableValue(desc="", user_default=0.5)

    def __post_init__(self):
        self._model = torch.jit.load(self.model)

    def check_init(self):
        pass

    async def discover_topics_sparse(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopicsSparse:
        raise NotImplementedError("Sparse topic discovery is not supported by CrossBertTopicer.")

    async def discover_topics_dense(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopics:
        raise NotImplementedError("Dense topic discovery is not supported by CrossBertTopicer.")
    
    async def discover_topics_in_db_sparse(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopicsSparse:
        raise NotImplementedError("Sparse topic discovery from DB is not supported by CrossBertTopicer.")

    async def discover_topics_in_db_dense(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopics:
        raise NotImplementedError("Dense topic discovery from DB is not supported by CrossBertTopicer.")

    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:
        """
        Propose tags for a given text chunk and return the proposals with span indices.
        
        :param text_chunk: A text chunk to propose tags for.
        :param tags: A list of tags to find in text_chunk.
        :return: TextChunkWithTagSpanProposals
        """
        ...

    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTagSpanProposals]:
        """
        Propose tags for texts found in a database based on a database request and return the proposals with span indices.
        
        :param tag: Tag to find in texts.
        :param db_request: Database request to fetch texts for tag proposal.
        :return: list[TextChunkWithTagSpanProposals]
        """
        if self.db_connection is None:
            raise MissingServiceError("DB connection is not set for CrossBertTopicer. This can happen if the class is not properly initialized.")
        
        text_chunks = self.db_connection.get_text_chunks(db_request)
        return [self.propose_tags(text_chunk, [tag]) for text_chunk in text_chunks]
