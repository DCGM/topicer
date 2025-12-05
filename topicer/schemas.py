from uuid import UUID
from pydantic import BaseModel


class TextChunk(BaseModel):
    id: UUID
    text: str


class Tag(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    # examples: list[TextWithSpan] | None = None


class TagSpanProposal(BaseModel):
    tag_id: UUID
    span_start: int
    span_end: int
    confidence: float | None = None
    reason: str | None = None


class TagSpanProposals(BaseModel):
    proposals: list[TagSpanProposal]


class TextChunkWithTagSpanProposals(TextChunk):
    tag_span_proposals: TagSpanProposals


class DBRequest(BaseModel):
    collection_id: UUID | None = None


class Topic(BaseModel):
    name: str
    name_explanation: str | None = None
    description: str | None = None
    # examples: list[TextWithSpan] | None = None


class DiscoveredTopics(BaseModel):
    topics: list[Topic]
    topic_documents: list[list[float]]  # N x K matrix of probabilities where N is number of topics and K is number of documents


class DiscoveredTopicsSparse(BaseModel):
    topics: list[Topic]
    topic_documents: list[
        list[tuple[int, float]]]  # N x K matrix where N is number of topics and K is number of documents, the tuple is (document_index, probability)
