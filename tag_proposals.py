from pydantic import BaseModel
from uuid import UUID



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
    tag_id: UUID
    span_start: int
    span_end: int
    confidence: float | None = None
    reason: str | None = None

class DBRequest(BaseModel):
    collection_id: UUID  | None = None

class DBSearch:
    pass
    def search_texts(self, query: str) -> list[str]:

class TextChunk(BaseModel):
    id: UUID
    text: str


class TextChunkWithTags(TextChunk):
    tags: list[TagSpanProposal]

"""
config: .yaml
- DB config 
-- db type
-- connection string
-- ...

api key, c

"""

class TagProposal:
    def __init__(self, config_file: str):
        pass

    def propose_tags(self, text: TextChunk, tag: list[Tag]) -> TextChunkWithTags:
        pass

    def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTags]:
        """

        """
        pass
