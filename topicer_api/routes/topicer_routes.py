import logging
from typing import Sequence
import asyncio
from fastapi import APIRouter, HTTPException, Query, status, Depends
from fastapi.responses import RedirectResponse, StreamingResponse

from topicer.llm.usage import collect_usage
from topicer.schemas import (DBRequest, DiscoveredTopics, DiscoveredTopicsSparse, Tag, TextChunk,
                             TextChunkWithTagSpanProposals, TextChunkWithTagSpanProposalsAndUsage)
from topicer_api.topicers import LoadedTopicers, get_loaded_topicers


logger = logging.getLogger(__name__)
topicer_router = APIRouter()


@topicer_router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@topicer_router.get("/configs", summary="List available Topicer configurations.")
async def get_configs(loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)) -> list[str]:
    configs = list(loaded_topicers.keys())
    return configs


@topicer_router.post("/topics/discover/texts/sparse", summary="Discover topics in provided texts using sparse approach.")
async def discover_topics_sparse(config_name: str, texts: Sequence[TextChunk], n: int | None = None, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)) -> DiscoveredTopicsSparse:
    if config_name not in loaded_topicers:
        logger.warning(f"Config {config_name} not found among loaded topicers: {list(loaded_topicers.keys())}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = loaded_topicers[config_name]
    logger.info(f"Using topicer config: {config_name}")

    try:
        result = await topicer_model.discover_topics_sparse(texts=texts, n=n)
        logger.info(f"Successfully discovered topics in texts using sparse method and config: {config_name}")
    except NotImplementedError:
        logger.warning(f"Sparse topic discovery in texts not implemented for config: {config_name}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/topics/discover/texts/dense", summary="Discover topics in provided texts using dense approach.")
async def discover_topics_dense(config_name: str, texts: Sequence[TextChunk], n: int | None = None, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)) -> DiscoveredTopics:
    if config_name not in loaded_topicers:
        logger.warning(f"Config {config_name} not found among loaded topicers: {list(loaded_topicers.keys())}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = loaded_topicers[config_name]
    logger.info(f"Using topicer config: {config_name}")

    try:
        result = await topicer_model.discover_topics_dense(texts=texts, n=n)
        logger.info(f"Successfully discovered topics in texts using dense method and config: {config_name}")
    except NotImplementedError:
        logger.warning(f"Dense topic discovery in texts not implemented for config: {config_name}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/topics/discover/db/sparse", summary="Discover topics in texts stored in database using sparse approach.")
async def discover_topics_in_db_sparse(config_name: str, db_request: DBRequest, n: int | None = None, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)) -> DiscoveredTopicsSparse:
    if config_name not in loaded_topicers:
        logger.warning(f"Config {config_name} not found among loaded topicers: {list(loaded_topicers.keys())}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = loaded_topicers[config_name]
    logger.info(f"Using topicer config: {config_name}")

    try:
        result = await topicer_model.discover_topics_in_db_sparse(db_request=db_request, n=n)
        logger.info(f"Successfully discovered topics in DB texts using sparse method and config: {config_name}")
    except NotImplementedError:
        logger.warning(f"Sparse topic discovery in DB texts not implemented for config: {config_name}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/topics/discover/db/dense", summary="Discover topics in texts stored in database using dense approach.")
async def discover_topics_in_db_dense(config_name: str, db_request: DBRequest, n: int | None = None, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)) -> DiscoveredTopics:
    if config_name not in loaded_topicers:
        logger.warning(f"Config {config_name} not found among loaded topicers: {list(loaded_topicers.keys())}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = loaded_topicers[config_name]
    logger.info(f"Using topicer config: {config_name}")

    try:
        result = await topicer_model.discover_topics_in_db_dense(db_request=db_request, n=n)
        logger.info(f"Successfully discovered topics in DB texts using dense method and config: {config_name}")
    except NotImplementedError:
        logger.warning(f"Dense topic discovery in DB texts not implemented for config: {config_name}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/tags/propose/texts", summary="Propose tags on provided text chunk.",
                     description="The optional `usage` field lists every underlying LLM provider attempt of the request.")
async def propose_tags(config_name: str, text_chunk: TextChunk, tags: list[Tag], loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)) -> TextChunkWithTagSpanProposalsAndUsage:
    if config_name not in loaded_topicers:
        logger.warning(f"Config {config_name} not found among loaded topicers: {list(loaded_topicers.keys())}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = loaded_topicers[config_name]
    logger.info(f"Using topicer config: {config_name}")

    try:
        with collect_usage() as usage:
            result = await topicer_model.propose_tags(text_chunk=text_chunk, tags=tags)
        logger.info(f"Successfully proposed tags in texts using config: {config_name}")
    except NotImplementedError:
        logger.warning(f"Tag proposal in texts not implemented for config: {config_name}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return TextChunkWithTagSpanProposalsAndUsage(**result.model_dump(), usage=usage.summary())


@topicer_router.post("/tags/propose/db", summary="Propose tags on texts stored in database.")
async def propose_tags_in_db(config_name: str, tag: Tag, db_request: DBRequest, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)) -> list[TextChunkWithTagSpanProposals]:
    if config_name not in loaded_topicers:
        logger.warning(f"Config {config_name} not found among loaded topicers: {list(loaded_topicers.keys())}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = loaded_topicers[config_name]
    logger.info(f"Using topicer config: {config_name}")

    try:
        result = await topicer_model.propose_tags_in_db(tag=tag, db_request=db_request)
        logger.info(f"Successfully proposed tags in DB texts using config: {config_name}")
    except NotImplementedError:
        logger.warning(f"Tag proposal in DB texts not implemented for config: {config_name}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post(
    "/tags/propose/texts/most_probable",
    summary="Find the single most probable tag for a text chunk using BERT zero-shot."
)
async def find_most_probable_tag(
    config_name: str,
    text_chunk: TextChunk,
    tags: list[Tag],
    loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)
) -> list[dict]:
    if config_name not in loaded_topicers:
        logger.warning(
            f"Config {config_name} not found among loaded topicers: {list(loaded_topicers.keys())}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Config {config_name} not found.")

    topicer_model = loaded_topicers[config_name]
    if not hasattr(topicer_model, "find_most_probable_tag"):
        logger.warning(
            f"Method 'find_most_probable_tag' not implemented for config: {config_name}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"Method not applicable to {config_name}.")

    logger.info(
        f"Using topicer config: {config_name} to find most probable tag")

    try:
        result = await asyncio.to_thread(
            topicer_model.find_most_probable_tag,
            text_chunk.text,
            tags
        )

        logger.info(
            f"Successfully found most probable tag using config: {config_name}")
        return result

    except Exception as e:
        logger.error(
            f"Error in find_most_probable_tag for config {config_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@topicer_router.post(
    "/tags/propose/db/stream",
    summary="Propose tags on texts from database, streaming results as they complete (NDJSON).",
    description=(
        "Streams `application/x-ndjson`, one JSON object per line. Every line except the last is a "
        "`TextChunkWithTagSpanProposals` (only chunks with proposals are sent). A normally completed stream ends with "
        "exactly one `TagProposalUsageSummary`, recognizable by `\"event\": \"usage_summary\"`, which is sent even "
        "when there are no proposals and reports the processing status, the applied chunk limit and the usage of "
        "every underlying LLM call. A stream without it was interrupted and its usage is unknown. "
        "Returns 422 when `max_chunks` is not positive or exceeds the config's safety ceiling."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            # One NDJSON line; TagProposalUsageSummary is registered in components by topicer_api.main.
            "content": {"application/x-ndjson": {"schema": {"oneOf": [
                {"$ref": "#/components/schemas/TextChunkWithTagSpanProposals"},
                {"$ref": "#/components/schemas/TagProposalUsageSummary"},
            ]}}},
            "description": "NDJSON stream of TextChunkWithTagSpanProposals lines terminated by one TagProposalUsageSummary line.",
        },
    },
)
async def propose_tags_in_db_stream(
        config_name: str,
        tag: Tag,
        db_request: DBRequest,
        max_chunks: int | None = Query(
            None, ge=1,
            description="Maximum number of candidate chunks to run inference on for this tag. When omitted, the "
                        "config's retrieval limit applies. Values above the config's `max_chunks_ceiling` are rejected with 422."),
        loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)) -> StreamingResponse:
    if config_name not in loaded_topicers:
        logger.warning(f"Config {config_name} not found among loaded topicers: {list(loaded_topicers.keys())}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = loaded_topicers[config_name]

    if not hasattr(topicer_model, "propose_tags_in_db_stream"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Streaming not supported for {config_name}.")

    # Validate before the stream starts, so an invalid limit never reaches inference.
    if max_chunks is not None and not hasattr(topicer_model, "validate_max_chunks"):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"max_chunks is not supported by {config_name}.")
    try:
        if max_chunks is not None:
            topicer_model.validate_max_chunks(max_chunks)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    async def generate():
        try:
            async for result in topicer_model.propose_tags_in_db_stream(tag=tag, db_request=db_request, max_chunks=max_chunks):
                yield result.model_dump_json() + "\n"
        except Exception as e:
            logger.error(f"Error during streaming tag proposal for config {config_name}: {e}")
            raise

    return StreamingResponse(generate(), media_type="application/x-ndjson")
