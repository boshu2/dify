from fastapi import APIRouter

from app.api.routes.agents import router as agents_router
from app.api.routes.chat import router as chat_router
from app.api.routes.datasources import router as datasources_router
from app.api.routes.meta import router as meta_router
from app.api.routes.providers import router as providers_router

api_router = APIRouter()

api_router.include_router(providers_router, prefix="/providers", tags=["providers"])
api_router.include_router(datasources_router, prefix="/datasources", tags=["datasources"])
api_router.include_router(agents_router, prefix="/agents", tags=["agents"])
api_router.include_router(meta_router, prefix="/meta", tags=["meta"])
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
