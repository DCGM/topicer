import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic.json_schema import models_json_schema

from topicer.base import BaseDBConnection
from topicer.schemas import TagProposalUsageSummary
from topicer_api.routes import topicer_router
from topicer_api.topicers import load_topicers


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded_topicers = load_topicers()
    logger.info("TopicerAPI started and topicer configurations loaded.")

    for topicer in app.state.loaded_topicers.values():
        if hasattr(topicer, "db_connection") and topicer.db_connection is not None and \
                hasattr(topicer.db_connection, "connect") and hasattr(topicer.db_connection, "close"):
            await topicer.db_connection.connect()
    yield

    for topicer in app.state.loaded_topicers.values():
        if hasattr(topicer, "db_connection") and topicer.db_connection is not None and \
                hasattr(topicer.db_connection, "connect") and hasattr(topicer.db_connection, "close"):
            await topicer.db_connection.close()

    logging.info("Shutting down TopicerAPI...")


app = FastAPI(title="TopicerAPI", lifespan=lifespan)
app.include_router(topicer_router, prefix="/v1")


def openapi():
    """OpenAPI schema including the models only referenced by the NDJSON stream response (not by a response_model)."""
    if app.openapi_schema is None:
        schema = FastAPI.openapi(app)
        _, defs = models_json_schema([(TagProposalUsageSummary, "serialization")],
                                     ref_template="#/components/schemas/{model}")
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, model_schema in defs.get("$defs", {}).items():
            components.setdefault(name, model_schema)
    return app.openapi_schema


app.openapi = openapi
