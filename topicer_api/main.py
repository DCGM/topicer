from fastapi import FastAPI

from topicer_api.routes import topicer_router

app = FastAPI()
app.include_router(topicer_router)
