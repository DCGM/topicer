import asyncio
import json
from contextlib import aclosing
from typing import AsyncGenerator
from topicer.base import BaseTopicer, MissingServiceError
from topicer.schemas import TextChunk, Tag, TagSpanProposal, TextChunkWithTagSpanProposals, TagProposalUsageSummary
from topicer.llm.usage import UsageCollector, collect_usage, describe_exception
from topicer.tagging.tagging_schemas import LLMTagProposalList
import logging
from topicer.schemas import DBRequest
from classconfig import ConfigurableMixin, ConfigurableValue
from classconfig.validators import MinValueIntegerValidator
from topicer.utils.fuzzy_matcher import FuzzyMatcher

class LLMTopicer(BaseTopicer, ConfigurableMixin):
    span_granularity: str = ConfigurableValue(
        desc="Granularity level for span extraction", user_default="phrase")
    max_chunks_ceiling: int = ConfigurableValue(
        desc="Safety ceiling for the per-request max_chunks of propose_tags_in_db_stream; larger requests are rejected.",
        user_default=200, voluntary=True, validator=MinValueIntegerValidator(1))
    max_concurrent_chunks: int = ConfigurableValue(
        desc="Maximum number of chunks processed concurrently by propose_tags_in_db_stream.",
        user_default=10, voluntary=True, validator=MinValueIntegerValidator(1))

    def __post_init__(self) -> None:
        self.fuzzy_matcher: FuzzyMatcher = FuzzyMatcher(max_dist_ratio=0.2)
        
    def check_init(self) -> None:
        """Check if all required services are set. Raise MissingServiceError if not."""
        if self.llm_service is None:
            raise MissingServiceError(
                "LLM service is not set for LLMTopicer.")

    async def propose_tags(self, text_chunk: TextChunk, tags: list[Tag]) -> TextChunkWithTagSpanProposals:

        tags_json = json.dumps([tag.model_dump(mode="json")
                               for tag in tags], ensure_ascii=False)

        # Prompt instructions for LLM
        instructions: str = f"""
        You are an expert data extraction assistant. Your task is to identify relevant sections in the text matching the provided tags.
        
        ### Instructions:
        1. Read the input text.
        2. Review the Available Tags. Each tag has a name and optionally a description (explaining what should and should not be annotated) and examples (representative strings that should be tagged). Use all available fields to understand the tag's meaning.
        3. Identify spans that match the definitions of the Available Tags.
        4. For each match, you MUST extract:
           - `quote`: The **EXACT** substring from the text. Copy it precisely, character for character.
           - `context_before`: The 5-10 words immediately preceding the quote, if available. This is crucial to locate the text if the phrase appears multiple times. Don't bother with whitespaces, just focus on words separated by spaces.
           - `context_after`: The 5-10 words immediately following the quote, if available. This is crucial to locate the text if the phrase appears multiple times. Don't bother with whitespaces, just focus on words separated by spaces.
           - `tag_id`: The id of the matching tag (copy the id exactly from the Available Tags list).
           - `confidence`: A score between 0.0 and 1.0. Indicate how confident you are that this quote matches the tag. Try to be as accurate as possible, the confidence doesn't necessarily need to be really close to 1.0. You don't need to only return high-confidence matches; lower-confidence matches are acceptable if you believe they might be relevant. It's better to provide more options for downstream processing, but do not flood with very low-confidence matches.
           - `reason`: (optional) A brief explanation of why you selected this quote for the tag. Always write the `reason` in the same language as the tags and their definitions (name/description/examples). If the language of the tags cannot be reliably determined, write the `reason` in the language of the document text being annotated.
        
        ### Constraints:
        - Do not paraphrase the quote.
        - Granularity level: {self.span_granularity}. Try to choose spans that fit this granularity approximately.
        - If no relevant spans are found for a tag, do not create any entries for it
        - The `reason` field MUST be written in the language of the tags/definitions; fall back to the document's language only if the tag language is unclear.
        """

        input_text: str = f"""
        ### Document Text:
        {text_chunk.text}
        
        ### Available Tags:
        {tags_json}
        """

        # Call LLM service to get structured tag proposals. The method expects a list of text chunks but we provide only one.
        llm_proposals_list: list[LLMTagProposalList] = await self.llm_service.process_text_chunks_structured(text_chunks=[input_text],
                                                                                                                instruction=instructions,
                                                                                                                output_type=LLMTagProposalList)

        # Extract proposals from the single response
        llm_proposals = llm_proposals_list[0].proposals

        final_proposals = []

        # Build a lookup map to resolve full Tag objects by id
        tags_by_id = {tag.id: tag for tag in tags}

        # Post-processing in Python (Calculating indices)
        for prop in llm_proposals:
            resolved_tag = tags_by_id.get(prop.tag_id)
            if resolved_tag is None:
                logging.warning(f"LLM returned unknown tag id '{prop.tag_id}', skipping.")
                continue

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
                        tag=resolved_tag,
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

    async def propose_tags_in_db(self, tag: Tag,  db_request: DBRequest) -> list[TextChunkWithTagSpanProposals]:
        if self.db_connection is None:
            raise MissingServiceError("DB connection is not set for LLMTopicer.")
        
        results: list[TextChunkWithTagSpanProposals] = []

        query_parts = [tag.name]
        if tag.description:
            query_parts.append(tag.description)
        if tag.examples:
            query_parts.extend(tag.examples)
        query_text = "\n".join(query_parts)

        tag_embedding = self.embedding_service.embed_queries([query_text])[0]

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

    def validate_max_chunks(self, max_chunks: int | None) -> None:
        """Raise ValueError when an explicit per-request max_chunks is not a positive integer within the safety ceiling."""
        if max_chunks is None:
            return
        if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1:
            raise ValueError(f"max_chunks must be a positive integer, got {max_chunks!r}.")
        if max_chunks > self.max_chunks_ceiling:
            raise ValueError(f"max_chunks={max_chunks} exceeds this configuration's limit of {self.max_chunks_ceiling}.")

    async def _propose_tags_bounded(self, text_chunks: list[TextChunk], tag: Tag, collector: UsageCollector
                                    ) -> AsyncGenerator[TextChunkWithTagSpanProposals | Exception, None]:
        """
        Run propose_tags on the chunks with at most `max_concurrent_chunks` tasks in flight, yielding results
        (or the exception of a failed chunk) as they complete. Outstanding tasks are cancelled and awaited
        when the generator is closed or cancelled.
        """
        async def run(chunk: TextChunk) -> TextChunkWithTagSpanProposals:
            with collect_usage(collector):
                return await self.propose_tags(text_chunk=chunk, tags=[tag])

        remaining = iter(text_chunks)
        pending: set[asyncio.Task] = set()

        def fill() -> None:
            while len(pending) < self.max_concurrent_chunks:
                chunk = next(remaining, None)
                if chunk is None:
                    return
                pending.add(asyncio.ensure_future(run(chunk)))

        try:
            fill()
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                fill()
                for task in done:
                    exc = task.exception()
                    yield exc if exc is not None else task.result()
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def propose_tags_in_db_stream(self, tag: Tag, db_request: DBRequest, max_chunks: int | None = None
                                        ) -> AsyncGenerator[TextChunkWithTagSpanProposals | TagProposalUsageSummary, None]:
        """
        Yield chunks with proposals as their LLM calls complete, then one TagProposalUsageSummary.

        :param max_chunks: Maximum number of candidate chunks to run inference on. When None, the database
            connection's configured retrieval limit (`chunks_limit`) applies.
        """
        if self.db_connection is None:
            raise MissingServiceError("DB connection is not set for LLMTopicer.")
        self.validate_max_chunks(max_chunks)

        if max_chunks is not None:
            limit, limit_source = max_chunks, "request"
        else:
            limit = getattr(self.db_connection, "chunks_limit", None)
            limit_source = "config" if limit is not None else None

        collector = UsageCollector()
        summary = TagProposalUsageSummary(status="complete", tag_id=tag.id, max_chunks=limit,
                                          max_chunks_source=limit_source)
        failed_chunks = 0
        delivered = False
        stage = "embedding"
        try:
            try:
                query_parts = [tag.name]
                if tag.description:
                    query_parts.append(tag.description)
                if tag.examples:
                    query_parts.extend(tag.examples)
                query_text = "\n".join(query_parts)

                tag_embedding = self.embedding_service.embed_queries([query_text])[0]

                # One extra candidate tells whether the limit actually excluded eligible chunks.
                stage = "retrieval"
                async with self.db_connection as db_conn:
                    text_chunks: list[TextChunk] = await db_conn.find_similar_text_chunks(
                        text=tag.name,
                        embedding=tag_embedding,
                        db_request=db_request,
                        k=limit + 1 if limit is not None else None
                    )

                if limit is not None and len(text_chunks) > limit:
                    summary.candidates_truncated = True
                    text_chunks = text_chunks[:limit]

                if not text_chunks:
                    logging.info(f"No similar text chunks found for tag '{tag.name}' in the database.")

                stage = "llm_inference"
                # aclosing: closing this stream (client disconnect) cancels the in-flight LLM calls immediately
                async with aclosing(self._propose_tags_bounded(text_chunks, tag, collector)) as results:
                    async for result in results:
                        summary.processed_chunks += 1
                        if isinstance(result, Exception):
                            failed_chunks += 1
                            logging.error(f"Tag proposal failed for a chunk of tag {tag.id} at stage llm_inference: "
                                          f"{describe_exception(result)}")
                        elif result.tag_span_proposals:
                            yield result

                if failed_chunks:
                    summary.status = "failed" if failed_chunks == summary.processed_chunks else "partial"
            except Exception as e:
                # No message or traceback: provider and DB errors may embed prompts or document text.
                logging.error(f"Tag proposal stream for tag {tag.id} failed at stage {stage}: {describe_exception(e)}")
                summary.status = "failed"

            summary.limit_reached = limit is not None and summary.processed_chunks >= limit
            summary.usage = collector.summary()
            yield summary
            delivered = True
        finally:
            if not delivered:
                # The client will not receive the summary (disconnect or cancellation): keep a server-side record.
                usage = collector.summary()
                known_cost = sum(c.cost_usd for c in usage.calls if c.cost_usd is not None)
                logging.warning(
                    f"Tag proposal stream for tag {tag.id} closed before its usage summary was delivered: "
                    f"processed_chunks={summary.processed_chunks}, attempted_calls={usage.attempted_calls}, "
                    f"completed_calls={usage.completed_calls}, known_cost_usd={known_cost}, "
                    f"calls_with_unknown_cost={sum(1 for c in usage.calls if c.cost_usd is None)}")
