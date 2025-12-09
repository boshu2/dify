from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class LLMProviderBase(BaseModel):
    name: str
    provider_type: ProviderType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLMProviderCreate(LLMProviderBase):
    pass


class LLMProviderUpdate(BaseModel):
    name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    is_active: Optional[bool] = None


class LLMProviderResponse(LLMProviderBase):
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    # Hide API key in responses
    api_key: Optional[str] = None

    class Config:
        from_attributes = True

    def model_post_init(self, __context) -> None:
        # Mask API key
        if self.api_key:
            object.__setattr__(self, 'api_key', "***hidden***")
