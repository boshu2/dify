import enum
import uuid

from sqlalchemy import Column, DateTime, Enum, String, Text
from sqlalchemy.sql import func

from app.core.database import Base


class DataSourceType(str, enum.Enum):
    FILE = "file"
    URL = "url"
    TEXT = "text"
    GITLAB = "gitlab"


class DataSource(Base):
    __tablename__ = "datasources"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    source_type = Column(Enum(DataSourceType), nullable=False)
    content = Column(Text, nullable=True)  # For text type or cached content
    source_path = Column(String(1000), nullable=True)  # File path, URL, or GitLab source path
    # GitLab-specific fields
    gitlab_url = Column(String(500), nullable=True)  # GitLab instance URL
    gitlab_token = Column(Text, nullable=True)  # Encrypted access token
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<DataSource {self.name} ({self.source_type})>"
