import json
from topicer.base import BaseTopicer
from topicer.schemas import TextChunk, Tag, TagSpanProposal, TextChunkWithTagSpanProposals
from topicer.tagging.tagging_schemas import LLMTagProposalList
import logging
from topicer.tagging.utils import find_exact_span
from topicer.database.db_schemas import DBRequest
from classconfig import ConfigurableMixin, ConfigurableValue
from topicer.base import OpenAIClientWrapper

class TagProposalV1(BaseTopicer, ConfigurableMixin):
    span_granularity: str = ConfigurableValue(desc="Granularity level for span extraction", user_default="phrase")
    
    @property
    def openai(self) -> OpenAIClientWrapper:
        return self.external_service  # nebo self.external_client, pokud nechceš .client

    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:

        tags_json = json.dumps([tag.model_dump(mode="json")
                               for tag in tags], ensure_ascii=False)

        # Prompt instructions for LLM
        instructions: str = f"""
        You are an expert data extraction assistant. Your task is to identify relevant sections in the text matching the provided tags.
        
        ### Instructions:
        1. Read the input text.
        2. Identify spans that match the definitions of the Available Tags.
        3. For each match, you MUST extract:
           - `quote`: The **EXACT** substring from the text. Copy it precisely, character for character.
           - `context_before`: The 5-10 words immediately preceding the quote. This is crucial to locate the text if the phrase appears multiple times.
           - `tag`: The matching tag object.
           - `confidence`: A score between 0.0 and 1.0. Indicate how confident you are that this quote matches the tag. Try to be as accurate as possible, the confidence doesn't necessarily need to be really close to 1.0. You don't need to only return high-confidence matches; lower-confidence matches are acceptable if you believe they might be relevant. It's better to provide more options for downstream processing, but do not flood with very low-confidence matches.
           - `reason`: (optional) A brief explanation of why you selected this quote for the tag.
        
        ### Constraints:
        - Do not paraphrase the quote.
        - Granularity level: {self.span_granularity}. Try to choose spans that fit this granularity approximately.
        - If no relevant spans are found for a tag, do not create any entries for it
        """

        input_text: str = f"""
        ### Document Text:
        {text_chunk.text}
        
        ### Available Tags:
        {tags_json}
        """

        response = await self.openai.client.responses.parse(
            model=self.openai.model,
            instructions=instructions,
            input=input_text,
            text_format=LLMTagProposalList,
            reasoning={
                "effort": self.openai.reasoning
            }
        )

        # it is already parsed as LLMTagProposalList
        llm_proposals = response.output_parsed.proposals

        final_proposals = []

        # Post-processing in Python (Calculating indices)
        for prop in llm_proposals:
            # We have quote and context_before, need to find indices in text_chunk.text
            indices = find_exact_span(
                text_chunk.text, prop.quote, prop.context_before)

            if indices:
                start, end = indices

                # We create the final object with indices that your application expects
                final_proposals.append(
                    TagSpanProposal(
                        tag=prop.tag,
                        span_start=start,
                        span_end=end,
                        confidence=prop.confidence,
                        reason=prop.reason
                    )
                )
            else:
                # Logging: LLM returned text that is not found in the document
                logging.warning(
                    f"Could not locate quote '{prop.quote}' in text chunk {text_chunk.id}")

        return TextChunkWithTagSpanProposals(
            id=text_chunk.id,
            text=text_chunk.text,
            tag_span_proposals=final_proposals
        )

    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTagSpanProposals]:
        pass