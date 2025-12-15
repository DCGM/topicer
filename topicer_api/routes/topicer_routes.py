import os
import logging
from typing import Sequence
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import RedirectResponse

from topicer.schemas import DBRequest, Tag, TextChunk

from topicer_api.topicers import LoadedTopicers, get_loaded_topicers


logger = logging.getLogger(__name__)
topicer_router = APIRouter()


@topicer_router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@topicer_router.get("/configs", summary="List available Topicer configurations.")
async def get_configs(loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)):
    configs = list(loaded_topicers.keys())
    return configs


@topicer_router.post("/topics/discover/texts/sparse", summary="Discover topics in provided texts using sparse approach.")
async def discover_topics_sparse(config_name: str, texts: Sequence[TextChunk], n: int | None = None, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)):
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
async def discover_topics_dense(config_name: str, texts: Sequence[TextChunk], n: int | None = None, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)):
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
async def discover_topics_in_db_sparse(config_name: str, db_request: DBRequest, n: int | None = None, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)):
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
async def discover_topics_in_db_dense(config_name: str, db_request: DBRequest, n: int | None = None, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)):
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


@topicer_router.post("/tags/propose/texts", summary="Propose tags on provided text chunk.")
async def propose_tags(config_name: str, text_chunk: TextChunk, tags: list[Tag], loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)):
    if config_name not in loaded_topicers:
        logger.warning(f"Config {config_name} not found among loaded topicers: {list(loaded_topicers.keys())}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = loaded_topicers[config_name]
    logger.info(f"Using topicer config: {config_name}")

    try:
        result = await topicer_model.propose_tags(text_chunk=text_chunk, tags=tags)
        logger.info(f"Successfully proposed tags in texts using config: {config_name}")
    except NotImplementedError:
        logger.warning(f"Tag proposal in texts not implemented for config: {config_name}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/tags/propose/db", summary="Propose tags on texts stored in database.")
async def propose_tags_in_db(config_name: str, tag: Tag, db_request: DBRequest, loaded_topicers: LoadedTopicers = Depends(get_loaded_topicers)):
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
