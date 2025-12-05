# THIS FILE CONTAINS PUBLIC SCHEMAS AVAILABLE FOR IMPORTING IN OTHER PACKAGES

from uuid import UUID
from pydantic import BaseModel


class Tag(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    # examples: list[TextWithSpan] | None = None


class TagSpanProposal(BaseModel):
    tag: Tag
    span_start: int
    span_end: int
    confidence: float | None = None
    reason: str | None = None


class TagSpanProposalList(BaseModel):
    proposals: list[TagSpanProposal]


class TextChunk(BaseModel):
    id: UUID
    text: str


class TextChunkWithTagSpanProposals(TextChunk):
    tag_span_proposals: list[TagSpanProposal]


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
    topic_documents: list[list[tuple[int, float]]]  # N x K matrix where N is number of topics and K is number of documents, the tuple is (document_index, probability)
