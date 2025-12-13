from typing import Sequence
from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from classconfig import Config

from topicer import factory
from topicer.schemas import DBRequest, Tag, TextChunk

topicer_router = APIRouter()


@topicer_router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@topicer_router.get("/configs")
async def get_configs():
    return []


@topicer_router.get("/discover_topics_sparse")
async def discover_topics_sparse(config: str | dict | Config, texts: Sequence[TextChunk], n: int | None = None):
    topicer_model = factory(config)

    try:
        result = await topicer_model.discover_topics_sparse(texts=texts, n=n)
    except NotImplementedError:
        result = {"message": "Not implemented"}

    return result


@topicer_router.get("/discover_topics_dense")
async def discover_topics_dense(config: str | dict | Config, texts: Sequence[TextChunk], n: int | None = None):
    topicer_model = factory(config)

    try:
        result = await topicer_model.discover_topics_dense(texts=texts, n=n)
    except NotImplementedError:
        result = {"message": "Not implemented"}

    return result


@topicer_router.get("/discover_topics_in_db_sparse")
async def discover_topics_in_db_sparse(config: str | dict | Config, db_request: DBRequest, n: int | None = None):
    topicer_model = factory(config)

    try:
        result = await topicer_model.discover_topics_in_db_sparse(db_request=db_request, n=n)
    except NotImplementedError:
        result = {"message": "Not implemented"}

    return result


@topicer_router.get("/discover_topics_in_db_dense")
async def discover_topics_in_db_dense(config: str | dict | Config, db_request: DBRequest, n: int | None = None):
    topicer_model = factory(config)

    try:
        result = await topicer_model.discover_topics_in_db_dense(db_request=db_request, n=n)
    except NotImplementedError:
        result = {"message": "Not implemented"}

    return result


@topicer_router.get("/propose_tags")
async def propose_tags(config: str | dict | Config, text_chunk: TextChunk, tags: list[Tag]):
    topicer_model = factory(config)

    try:
        result = await topicer_model.propose_tags(text_chunk=text_chunk, tags=tags)
    except NotImplementedError:
        result = {"message": "Not implemented"}

    return result


@topicer_router.get("/propose_tags_in_db")
async def propose_tags_in_db(config: str | dict | Config, tag: Tag, db_request: DBRequest):
    topicer_model = factory(config)

    try:
        result = await topicer_model.propose_tags_in_db(tag=tag, db_request=db_request)
    except NotImplementedError:
        result = {"message": "Not implemented"}

    return result
