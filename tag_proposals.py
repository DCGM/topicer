import yaml
import json
from typing import List
from uuid import UUID, uuid4
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
# import weaviate

# weaviate_client = weaviate.connect_to_local(
#     host="localhost",
#     port=9000,
#     grpc_port=50055
# )

# print(weaviate_client.is_ready())


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
    def __init__(self, config_file: str, open_ai_client: OpenAI):
        self.config = Config(config_file)
        self.open_ai_client = open_ai_client

    async def propose_tags(self, text_chunk: TextChunk, tag: list[Tag]) -> TextChunkWithTags:
        response = self.open_ai_client.responses.parse(
            model="gpt-5-nano",
            instructions="You are an expert tag proposer. Given the text, suggest relevant tags from the provided list.",
            input=text_chunk.text,
            text_format=TagSpanProposal
        )
        print(response.output_text)
        return TextChunkWithTags(id=text_chunk.id, text=text_chunk.text, tags=[])

    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTags]:
        return []
    


if __name__ == "__main__":
    import asyncio
    from test_data import text_chunk, tag1, tag2, tag3

    load_dotenv()

    API_KEY = os.getenv("OPENAI_API_KEY")
    
    print("API Key loaded" if API_KEY is not None else "API Key not found")

    open_ai_client = OpenAI(
        api_key=API_KEY,
    )

    tag_proposal = TagProposal("config.yaml", open_ai_client)
    # propose_tags is async; run it with asyncio
    asyncio.run(tag_proposal.propose_tags(text_chunk, [tag1, tag2, tag3]))
