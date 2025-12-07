from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from app.schemas.provider import LLMProviderResponse
from app.schemas.datasource import DataSourceResponse


class AgentBase(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: str


class AgentCreate(AgentBase):
    provider_id: str
    datasource_ids: list[str] = []


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    provider_id: Optional[str] = None
    datasource_ids: Optional[list[str]] = None


class AgentResponse(AgentBase):
    id: str
    provider_id: str
    provider: LLMProviderResponse
    datasources: list[DataSourceResponse] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatMessage(BaseModel):
    role: str  # user, assistant
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    usage: Optional[dict] = None
