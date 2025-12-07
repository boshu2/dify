"""
Pytest configuration and shared fixtures for LiteAgent tests.
"""
import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app


# Test database URL - in-memory SQLite
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    # Enable foreign keys for SQLite
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def test_app(db_session: AsyncSession) -> FastAPI:
    """Create a test FastAPI application with overridden dependencies."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield app
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create a test HTTP client."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as ac:
        yield ac


# ============== Factory Fixtures ==============

@pytest.fixture
def provider_data() -> dict[str, Any]:
    """Factory for creating provider test data."""
    def _provider_data(
        name: str = "Test Provider",
        provider_type: str = "openai",
        model_name: str = "gpt-4o",
        api_key: str = "sk-test-key-123",
        base_url: str | None = None,
    ) -> dict[str, Any]:
        data = {
            "name": name,
            "provider_type": provider_type,
            "model_name": model_name,
            "api_key": api_key,
        }
        if base_url:
            data["base_url"] = base_url
        return data
    return _provider_data


@pytest.fixture
def datasource_data() -> dict[str, Any]:
    """Factory for creating datasource test data."""
    def _datasource_data(
        name: str = "Test DataSource",
        source_type: str = "text",
        content: str = "This is test content for the datasource.",
        source_path: str | None = None,
    ) -> dict[str, Any]:
        data = {
            "name": name,
            "source_type": source_type,
        }
        if source_type == "text":
            data["content"] = content
        elif source_type == "url":
            data["source_path"] = source_path or "https://example.com"
        return data
    return _datasource_data


@pytest.fixture
def agent_data() -> dict[str, Any]:
    """Factory for creating agent test data."""
    def _agent_data(
        name: str = "Test Agent",
        description: str = "A test agent",
        system_prompt: str = "You are a helpful assistant.",
        provider_id: str = "",
        datasource_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "provider_id": provider_id,
            "datasource_ids": datasource_ids or [],
        }
    return _agent_data


# ============== Mock Fixtures ==============

@pytest.fixture
def mock_openai_response() -> dict[str, Any]:
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the mock.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_anthropic_response() -> dict[str, Any]:
    """Mock Anthropic API response."""
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "This is a test response from the mock.",
            }
        ],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
        },
    }
