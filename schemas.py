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