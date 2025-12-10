"""
Sandbox service main entry point.

A standalone microservice for safe code execution.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_config
from app.executor import get_executor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    config = get_config()
    logger.info(f"Starting sandbox service on port {config.port}")
    logger.info(f"Max workers: {config.max_workers}")
    logger.info(f"Execution timeout: {config.execution_timeout}s")
    logger.info(f"Network enabled: {config.enable_network}")

    # Initialize executor
    get_executor()

    yield

    # Cleanup
    executor = get_executor()
    await executor.close()
    logger.info("Sandbox service shutdown")


# Create FastAPI app
app = FastAPI(
    title="LiteAgent Sandbox Service",
    description="Isolated code execution service for LiteAgent",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


def main():
    """Run the sandbox service."""
    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        workers=1,  # Use single worker, executor handles parallelism
        log_level="info" if not config.debug else "debug",
    )


if __name__ == "__main__":
    main()
