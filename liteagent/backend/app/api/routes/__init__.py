from fastapi import APIRouter

from app.api.routes.agents import router as agents_router
from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.datasources import router as datasources_router
from app.api.routes.health import router as health_router
from app.api.routes.meta import router as meta_router
from app.api.routes.providers import router as providers_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.sandbox import router as sandbox_router
from app.api.routes.traces import router as traces_router
from app.api.routes.workflows import router as workflows_router

api_router = APIRouter()

# Health check routes (no prefix for /health, /ping, etc.)
api_router.include_router(health_router)

# API routes
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(providers_router, prefix="/providers", tags=["providers"])
api_router.include_router(datasources_router, prefix="/datasources", tags=["datasources"])
api_router.include_router(agents_router, prefix="/agents", tags=["agents"])
api_router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
api_router.include_router(meta_router, prefix="/meta", tags=["meta"])
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(workflows_router, prefix="/workflows", tags=["workflows"])
api_router.include_router(retrieval_router, prefix="/retrieval", tags=["retrieval"])
api_router.include_router(sandbox_router, tags=["sandbox"])
api_router.include_router(traces_router, tags=["traces"])
