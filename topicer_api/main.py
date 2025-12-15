import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from topicer_api.routes import topicer_router
from topicer_api.topicers import load_topicers


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded_topicers = load_topicers()
    logger.info("Topicer API started and topicer configurations loaded.")
    yield
    logging.info("Shutting down Topicer API...")


app = FastAPI(lifespan=lifespan)
app.include_router(topicer_router, prefix="/v1")
