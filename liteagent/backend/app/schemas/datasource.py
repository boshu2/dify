from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Optional


class DataSourceType(str, Enum):
    FILE = "file"
    URL = "url"
    TEXT = "text"
    GITLAB = "gitlab"


class DataSourceBase(BaseModel):
    name: str
    source_type: DataSourceType


class DataSourceCreate(DataSourceBase):
    content: Optional[str] = None  # For text type
    source_path: Optional[str] = None  # For URL or GitLab source path
    # GitLab-specific fields
    gitlab_url: Optional[str] = None  # e.g., https://gitlab.com
    gitlab_token: Optional[str] = None  # Personal access token


class DataSourceUpdate(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None
    source_path: Optional[str] = None
    gitlab_token: Optional[str] = None  # Allow updating the token


class DataSourceResponse(DataSourceBase):
    id: str
    content: Optional[str] = None
    source_path: Optional[str] = None
    gitlab_url: Optional[str] = None
    # Token is hidden in responses
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
