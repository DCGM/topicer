import asyncio
import json
from topicer.base import BaseTopicer, MissingServiceError
from topicer.schemas import TextChunk, Tag, TagSpanProposal, TextChunkWithTagSpanProposals
from topicer.tagging.tagging_schemas import LLMTagProposalList
import logging
from topicer.schemas import DBRequest
from classconfig import ConfigurableMixin, ConfigurableValue
from topicer.utils.fuzzy_matcher import FuzzyMatcher

class LLMTopicer(BaseTopicer, ConfigurableMixin):
    span_granularity: str = ConfigurableValue(
        desc="Granularity level for span extraction", user_default="phrase")

    def __post_init__(self) -> None:
        self.fuzzy_matcher: FuzzyMatcher = FuzzyMatcher(max_dist_ratio=0.2)
        
    def check_init(self) -> None:
        """Check if all required services are set. Raise MissingServiceError if not."""
        if self.llm_service is None:
            raise MissingServiceError(
                "LLM service is not set for LLMTopicer.")
        if self.db_connection is None:
            raise MissingServiceError(
                "DB connection is not set for LLMTopicer.")

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
           - `context_before`: The 5-10 words immediately preceding the quote, if available. This is crucial to locate the text if the phrase appears multiple times. Don't bother with whitespaces, just focus on words separated by spaces.
           - `context_after`: The 5-10 words immediately following the quote, if available. This is crucial to locate the text if the phrase appears multiple times. Don't bother with whitespaces, just focus on words separated by spaces.
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

        async with self.llm_service as llm_service:
            # Call LLM service to get structured tag proposals. The method expects a list of text chunks but we provide only one.
            llm_proposals_list: list[LLMTagProposalList] = await llm_service.process_text_chunks_structured(text_chunks=[input_text],
                                                                                                            instruction=instructions,
                                                                                                            output_type=LLMTagProposalList)

        # Extract proposals from the single response
        llm_proposals = llm_proposals_list[0].proposals

        final_proposals = []

        # Post-processing in Python (Calculating indices)
        for prop in llm_proposals:
            # We have quote and context_before, need to find indices in text_chunk.text
            indices = self.fuzzy_matcher.find_best_span(
                full_text=text_chunk.text,
                quote=prop.quote,
                context_before=prop.context_before,
                context_after=prop.context_after
            )

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
                    f"Could not locate quote '{prop.quote}' in text chunk {text_chunk.id}"
                    f" using provided context_before '{prop.context_before}' and context_after '{prop.context_after}'."
                )

        return TextChunkWithTagSpanProposals(
            id=text_chunk.id,
            text=text_chunk.text,
            tag_span_proposals=final_proposals
        )

    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest | None) -> list[TextChunkWithTagSpanProposals]:
        # if db_request is None:
        #     raise ValueError(
        #         "DB request must be provided for propose_tags_in_db in LLMTopicer.")

        results: list[TextChunkWithTagSpanProposals] = []

        tag_embedding = self.embedding_service.embed_queries([tag.name])[0]

        async with self.db_connection as db_conn:
            text_chunks: list[TextChunk] = await db_conn.find_similar_text_chunks(
                text=tag.name,
                embedding=tag_embedding,
                db_request=db_request
            )

        if not text_chunks:
            logging.info(
                f"No similar text chunks found for tag '{tag.name}' in the database.")
            return results

        tasks = [self.propose_tags(text_chunk=chunk, tags=[tag])
                for chunk in text_chunks]
        all_proposals = await asyncio.gather(*tasks)

        return [p for p in all_proposals if p.tag_span_proposals]