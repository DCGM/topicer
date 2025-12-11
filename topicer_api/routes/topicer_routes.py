from fastapi import APIRouter
from fastapi.responses import RedirectResponse


topicer_router = APIRouter()


@topicer_router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
