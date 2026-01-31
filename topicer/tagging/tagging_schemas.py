# THIS FILE CONTAINS PRIVATE SCHEMAS USED FOR TAGGING PROPOSALS

from pydantic import BaseModel, Field
from topicer.schemas import Tag

# Tagging schemas


class LLMTagProposal(BaseModel):
    """ LLM proposal for tagging a text with a specific tag """
    tag: Tag
    quote: str = Field(...,
                       description="The exact substring from the text that corresponds to the tag.")
    context_before: str | None = Field(
        None, description="The 5-10 words immediately preceding the quote to help identify the location uniquely.")
    context_after: str | None = Field(
        None, description="The 5-10 words immediately following the quote if available.")
    confidence: float
    reason: str | None = None

class LLMTagProposalList(BaseModel):
    """ List of LLM tag proposals """
    proposals: list[LLMTagProposal]
