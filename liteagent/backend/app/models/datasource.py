from sqlalchemy import Column, String, Text, DateTime, Enum
from sqlalchemy.sql import func
import uuid
import enum

from app.core.database import Base


class DataSourceType(str, enum.Enum):
    FILE = "file"
    URL = "url"
    TEXT = "text"


class DataSource(Base):
    __tablename__ = "datasources"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    source_type = Column(Enum(DataSourceType), nullable=False)
    content = Column(Text, nullable=True)  # For text type or cached content
    source_path = Column(String(1000), nullable=True)  # File path or URL
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<DataSource {self.name} ({self.source_type})>"
