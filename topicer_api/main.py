import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from topicer.base import BaseDBConnection
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
