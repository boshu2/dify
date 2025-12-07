from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from typing import Optional


class DataSourceType(str, Enum):
    FILE = "file"
    URL = "url"
    TEXT = "text"


class DataSourceBase(BaseModel):
    name: str
    source_type: DataSourceType


class DataSourceCreate(DataSourceBase):
    content: Optional[str] = None  # For text type
    source_path: Optional[str] = None  # For URL type


class DataSourceUpdate(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None


class DataSourceResponse(DataSourceBase):
    id: str
    content: Optional[str] = None
    source_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
