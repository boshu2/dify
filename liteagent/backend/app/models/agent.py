import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Table, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base

# Many-to-many relationship between agents and datasources
agent_datasources = Table(
    "agent_datasources",
    Base.metadata,
    Column("agent_id", String(36), ForeignKey("agents.id"), primary_key=True),
    Column("datasource_id", String(36), ForeignKey("datasources.id"), primary_key=True),
)


class Agent(Base):
    __tablename__ = "agents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=False)
    provider_id = Column(String(36), ForeignKey("llm_providers.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    provider = relationship("LLMProvider", lazy="joined")
    datasources = relationship("DataSource", secondary=agent_datasources, lazy="joined")

    def __repr__(self) -> str:
        return f"<Agent {self.name}>"
