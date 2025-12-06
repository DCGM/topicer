# THIS FILE CONTAINS PUBLIC SCHEMAS AVAILABLE FOR IMPORTING IN OTHER PACKAGES

from uuid import UUID
from pydantic import BaseModel, Field

class TextWithSpan(BaseModel):
    text: str
    span_start: int | None = None
    span_end: int | None = None

class Tag(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    examples: list[TextWithSpan] | None = None
    
class TagSpanProposal(BaseModel):
    tag: Tag
    span_start: int
    span_end: int
    confidence: float | None = None
    reason: str | None = None

class TextChunk(BaseModel):
    id: UUID
    text: str
    
class TextChunkWithTagSpanProposals(TextChunk):
    tag_span_proposals: list[TagSpanProposal]