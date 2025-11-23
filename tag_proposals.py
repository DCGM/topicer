import yaml
import json
from typing import List
from uuid import UUID
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
    collection_id: UUID | None = None


class DBSearch:
    pass


class TextChunk(BaseModel):
    id: UUID
    text: str


class TextChunkWithTags(TextChunk):
    tags: list[TagSpanProposal]


class Config:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        print('config: ' + json.dumps(self.cfg, indent=2))


class TagProposal:
    def __init__(self, config_file: str):
        self.config = Config(config_file)

    async def propose_tags(self, text: TextChunk, tag: list[Tag]) -> TextChunkWithTags:
        return TextChunkWithTags(id=text.id, text=text.text, tags=[])

    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTags]:
        return []


if __name__ == "__main__":
    tag_proposal = TagProposal("config.yaml")