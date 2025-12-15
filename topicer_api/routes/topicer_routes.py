import os
from typing import Sequence
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse

from topicer import factory
from topicer.schemas import DBRequest, Tag, TextChunk

from topicer_api.config import config as app_config


topicer_router = APIRouter()


@topicer_router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@topicer_router.get("/configs", summary="List available Topicer configurations.")
async def get_configs():
    configs = [os.path.splitext(file)[0] for file in os.listdir(app_config.TOPICER_API_CONFIGS_DIR)
               if file.endswith(app_config.TOPICER_API_CONFIGS_EXTENSION)]
    return configs


@topicer_router.post("/topics/discover/texts/sparse", summary="Discover topics in provided texts using sparse approach.")
async def discover_topics_sparse(config_name: str, texts: Sequence[TextChunk], n: int | None = None):
    config_path = str(os.path.join(app_config.TOPICER_API_CONFIGS_DIR, config_name + app_config.TOPICER_API_CONFIGS_EXTENSION))
    if not os.path.exists(config_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = factory(config_path)

    try:
        result = await topicer_model.discover_topics_sparse(texts=texts, n=n)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/topics/discover/texts/dense", summary="Discover topics in provided texts using dense approach.")
async def discover_topics_dense(config_name: str, texts: Sequence[TextChunk], n: int | None = None):
    config_path = str(os.path.join(app_config.TOPICER_API_CONFIGS_DIR, config_name + app_config.TOPICER_API_CONFIGS_EXTENSION))
    if not os.path.exists(config_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = factory(config_path)

    try:
        result = await topicer_model.discover_topics_dense(texts=texts, n=n)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/topics/discover/db/sparse", summary="Discover topics in texts stored in database using sparse approach.")
async def discover_topics_in_db_sparse(config_name: str, db_request: DBRequest, n: int | None = None):
    config_path = str(os.path.join(app_config.TOPICER_API_CONFIGS_DIR, config_name + app_config.TOPICER_API_CONFIGS_EXTENSION))
    if not os.path.exists(config_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = factory(config_path)

    try:
        result = await topicer_model.discover_topics_in_db_sparse(db_request=db_request, n=n)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/topics/discover/db/dense", summary="Discover topics in texts stored in database using dense approach.")
async def discover_topics_in_db_dense(config_name: str, db_request: DBRequest, n: int | None = None):
    config_path = str(os.path.join(app_config.TOPICER_API_CONFIGS_DIR, config_name + app_config.TOPICER_API_CONFIGS_EXTENSION))
    if not os.path.exists(config_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = factory(config_path)

    try:
        result = await topicer_model.discover_topics_in_db_dense(db_request=db_request, n=n)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/tags/propose/texts", summary="Propose tags on provided text chunk.")
async def propose_tags(config_name: str, text_chunk: TextChunk, tags: list[Tag]):
    config_path = str(os.path.join(app_config.TOPICER_API_CONFIGS_DIR, config_name + app_config.TOPICER_API_CONFIGS_EXTENSION))
    if not os.path.exists(config_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = factory(config_path)

    try:
        result = await topicer_model.propose_tags(text_chunk=text_chunk, tags=tags)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result


@topicer_router.post("/tags/propose/db", summary="Propose tags on texts stored in database.")
async def propose_tags_in_db(config_name: str, tag: Tag, db_request: DBRequest):
    config_path = str(os.path.join(app_config.TOPICER_API_CONFIGS_DIR, config_name + app_config.TOPICER_API_CONFIGS_EXTENSION))
    if not os.path.exists(config_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Config {config_name} not found.")

    topicer_model = factory(config_path)

    try:
        result = await topicer_model.propose_tags_in_db(tag=tag, db_request=db_request)
    except NotImplementedError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Method not applicable to {config_name}.")

    return result
