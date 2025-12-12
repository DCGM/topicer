from typing import Sequence

from classconfig import ConfigurableMixin, ConfigurableValue
from gliner import GLiNER

from topicer.base import BaseTopicer, MissingServiceError
from topicer.schemas import DiscoveredTopicsSparse, TagSpanProposal, TextChunk, DiscoveredTopics, DBRequest, Tag, TextChunkWithTagSpanProposals


class GlinerTopicer(BaseTopicer, ConfigurableMixin):
    model: str = ConfigurableValue(desc="Either a GLiNER model name from Hugging Face or a local path to a finetuned GLiNER.", user_default="urchade/gliner_multi-v2.1")
    threshold: float = ConfigurableValue(desc="", user_default=0.5)
    multi_label: bool = ConfigurableValue(desc="", user_default=False)

    def __post_init__(self):
        self._model = GLiNER.from_pretrained(self.model)

    def check_init(self):
        pass
        # if self.db_connection is None:
        #     raise MissingServiceError("DB connection has to be set for GlinerTopicer.")

    async def discover_topics_sparse(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopicsSparse:
        raise NotImplementedError("Sparse topic discovery is not supported by GlinerTopicer.")

    async def discover_topics_dense(self, texts: Sequence[TextChunk], n: int | None = None) -> DiscoveredTopics:
        raise NotImplementedError("Dense topic discovery is not supported by GlinerTopicer.")
    
    async def discover_topics_in_db_sparse(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopicsSparse:
        raise NotImplementedError("Sparse topic discovery from DB is not supported by this Topicer.")

    async def discover_topics_in_db_dense(self, db_request: DBRequest, n: int | None = None) -> DiscoveredTopics:
        raise NotImplementedError("Dense topic discovery from DB is not supported by this Topicer.")

    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:
        """
        Propose tags for a given text chunk and return the proposals with span indices.
        
        :param text_chunk: A text chunk to propose tags for.
        :param tags: A list of tags to find in text_chunk.
        :return: TextChunkWithTagSpanProposals
        """
        
        text = text_chunk.text
        topics = [tag.name for tag in tags]
        model_outputs = self._model.predict_entities(
            text,
            topics,
            threshold=self.threshold,
            multi_label=self.multi_label,
        )

        result = TextChunkWithTagSpanProposals(
            id=text_chunk.id,
            text=text_chunk.text,
            tag_span_proposals=[
                TagSpanProposal(
                    tag=next(tag for tag in tags if tag.name == output["label"]),
                    span_start=output["start"],
                    span_end=output["end"],
                    confidence=output["score"],
                    reason=None,
                )
                for output in model_outputs
            ]
        )
        return result

    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTagSpanProposals]:
        """
        Propose tags for texts found in a database based on a database request and return the proposals with span indices.
        
        :param tag: Tag to find in texts.
        :param db_request: Database request to fetch texts for tag proposal.
        :return: list[TextChunkWithTagSpanProposals]
        """
        if self.db_connection is None:
            raise MissingServiceError("DB connection is not set for GlinerTopicer. This can happen if the Topicer is not properly initialized.")
        
        text_chunks = self.db_connection.get_text_chunks(db_request)
        return [self.propose_tags(text_chunk, [tag]) for text_chunk in text_chunks]
