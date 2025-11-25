import yaml
import json
from typing import List
from uuid import UUID, uuid4
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
from schemas import TextChunk, Tag, TagSpanProposal, TagSpanProposals, TextChunkWithTagSpanProposals

# class TextWithSpan(BaseModel):
#     text: str
#     span_start: int | None = None
#     span_end: int | None = None

class DBRequest(BaseModel):
    collection_id: UUID | None = None

class DBSearch:
    pass

class Config:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            cfg: dict = yaml.safe_load(f)
        # self.cfg = cfg
        # print('config: ' + json.dumps(self.cfg, indent=2))
        self.weaviate_cfg: dict = cfg.get('weaviate', {})
        self.openai_cfg: dict = cfg.get('openai', {})


class TagProposal:
    def __init__(self, config_file: str, openai_client: OpenAI):
        self.config = Config(config_file)
        self.openai_client = openai_client

    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:
        tags_json = json.dumps([tag.model_dump(mode="json") for tag in tags])
        instructions: str = "You are an expert tag proposer. Given the text, suggest relevant tags from the provided list. You must provide the tag ID, the start and end span of the text that corresponds to the tag, and a confidence score between 0 and 1. If applicable, provide a brief reason for your choice. The number of spans each tag can correspond to is not given. One tag can correspond to 0, 1 or many spans in the text, it's your job to find all the relevant passages for the tag. Respond in JSON format."
        input: str = text_chunk.text + "\n\nAvailable tags: \n" + tags_json
        
        response = self.openai_client.responses.parse(
            model=self.config.openai_cfg.get('model', 'gpt-5-mini'),
            instructions=instructions,
            input=input,
            text_format=TagSpanProposals,
            reasoning={
                "effort":self.config.openai_cfg.get('reasoning', 'medium')
            }
        )
        
        try:
            parsed_proposals = TagSpanProposals.model_validate_json(response.output_text)
        except Exception as e:
            print(f"Error parsing TagSpanProposals: {e}")
            parsed_proposals = TagSpanProposals(proposals=[])            
        return TextChunkWithTagSpanProposals(id=text_chunk.id, text=text_chunk.text, tag_span_proposals=parsed_proposals)

    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTagSpanProposals]:
        return []


if __name__ == "__main__":
    import asyncio
    from test_data import text_chunk, tag1, tag2, tag3

    load_dotenv()

    API_KEY = os.getenv("OPENAI_API_KEY")

    print("API Key loaded" if API_KEY is not None else "API Key not found")

    openai_client = OpenAI(
        api_key=API_KEY,
    )
    
    # weaviate_client = weaviate.connect_to_local(
    #     host="localhost",
    #     port=9000,
    #     grpc_port=50055
    # )

    # print(weaviate_client.is_ready())

    tag_proposal = TagProposal("config.yaml", openai_client)
    # propose_tags is async; run it with asyncio
    asyncio.run(tag_proposal.propose_tags(text_chunk, [tag1, tag2, tag3]))
