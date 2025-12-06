# THIS FILE CONTAINS PRIVATE SCHEMAS USED FOR TAGGING PROPOSALS

from pydantic import BaseModel, Field
from topicer.schemas import Tag

# Tagging schemas
class LLMTagProposal(BaseModel):
    """ LLM proposal for tagging a text with a specific tag """
    tag: Tag
    quote: str = Field(..., description="The exact substring from the text that corresponds to the tag.")
    context_before: str = Field(..., description="The 5-10 words immediately preceding the quote to help identify the location uniquely.")
    confidence: float
    reason: str | None = None
    
class LLMTagProposalList(BaseModel):
    """ List of LLM tag proposals """
    proposals: list[LLMTagProposal]
    
# Config schemas

class WeaviateCfg(BaseModel):
    host: str = Field(default="localhost")
    rest_port: int = Field(default=8080)
    grpc_port: int = Field(default=50051)

class OpenAICfg(BaseModel):
    model: str = Field(default="gpt-4o")
    reasoning: str = Field(default="medium")
    span_granularity: str = Field(default="phrase")

class AppConfig(BaseModel):
    weaviate: WeaviateCfg
    openai: OpenAICfg