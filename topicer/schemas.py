# THIS FILE CONTAINS PUBLIC SCHEMAS AVAILABLE FOR IMPORTING IN OTHER PACKAGES

from typing import Literal
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
    examples: list[str] | None = None

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


class DBRequest(BaseModel):
    collection_id: UUID | None = None
    document_id: UUID | None = None


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


# --- LLM usage accounting ---
#
# usage_source: where input/output token counts come from.
#   "reported"  - both counts were returned by the provider
#   "estimated" - reserved; Topicer currently never estimates token counts
#   "unknown"   - at least one count is unavailable (failed or cancelled call, provider did not report usage)
# cost_source: where cost_usd comes from.
#   "reported"  - the provider returned the cost (e.g. OpenRouter usage.cost)
#   "estimated" - computed from reported token counts and the server-configured model prices
#   "unknown"   - cost_usd is null; it is NOT zero, the call may have been billed
UsageSource = Literal["reported", "estimated", "unknown"]
LLMCallOutcome = Literal["success", "error", "cancelled"]


class LLMCallUsage(BaseModel):
    """Usage of one underlying provider attempt (a retried request is several attempts)."""
    provider: str
    model: str | None = None
    outcome: LLMCallOutcome
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage_source: UsageSource = "unknown"
    cost_usd: float | None = None
    cost_source: UsageSource = "unknown"


class LLMUsage(BaseModel):
    attempted_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0  # error or cancelled
    calls: list[LLMCallUsage] = Field(default_factory=list)

    @classmethod
    def from_calls(cls, calls: list[LLMCallUsage]) -> "LLMUsage":
        completed = sum(1 for c in calls if c.outcome == "success")
        return cls(attempted_calls=len(calls), completed_calls=completed,
                   failed_calls=len(calls) - completed, calls=list(calls))


class TextChunkWithTagSpanProposalsAndUsage(TextChunkWithTagSpanProposals):
    """Response of POST /v1/tags/propose/texts: the proposals plus the LLM usage of the request."""
    usage: LLMUsage | None = None


class TagProposalUsageSummary(BaseModel):
    """
    Terminal event of the NDJSON stream of POST /v1/tags/propose/db/stream.

    Clients distinguish it from chunk results by `event == "usage_summary"`; chunk results have no `event` field.

    status:
      "complete" - every processed chunk was handled without error
      "partial"  - some processed chunks failed (their calls are still listed in usage)
      "failed"   - all processed chunks failed, or retrieval failed before any processing
    """
    event: Literal["usage_summary"] = "usage_summary"
    status: Literal["complete", "partial", "failed"]
    tag_id: UUID
    max_chunks: int | None = Field(
        None, description="Effective per-request limit on processed chunks (requested or config-driven); null when unbounded.")
    max_chunks_source: Literal["request", "config"] | None = None
    processed_chunks: int = Field(
        0, description="Chunks for which inference was attempted, including empty results and failures.")
    limit_reached: bool = False
    candidates_truncated: bool = Field(
        False, description="True when retrieval found more eligible candidates than max_chunks.")
    usage: LLMUsage = Field(default_factory=LLMUsage)
