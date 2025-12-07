from sqlalchemy import Column, String, Boolean, Text, DateTime
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class LLMProvider(Base):
    __tablename__ = "llm_providers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    provider_type = Column(String(50), nullable=False)  # openai, anthropic, ollama
    api_key = Column(Text, nullable=True)  # encrypted in production
    base_url = Column(String(500), nullable=True)
    model_name = Column(String(100), nullable=False)  # gpt-4, claude-3-opus, llama2
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<LLMProvider {self.name} ({self.provider_type}/{self.model_name})>"
