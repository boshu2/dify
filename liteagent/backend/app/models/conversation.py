"""Conversation and message models for chat history."""
import uuid
from enum import Enum

from sqlalchemy import Column, DateTime, Enum as SQLEnum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class MessageRole(str, Enum):
    """Role of message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    """Conversation/chat session model."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=True, index=True)  # Nullable for anonymous
    agent_id = Column(String(36), ForeignKey("agents.id"), nullable=False, index=True)
    title = Column(String(200), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    agent = relationship("Agent", lazy="joined")
    messages = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.created_at",
        lazy="dynamic",
    )

    def __repr__(self) -> str:
        return f"<Conversation {self.id[:8]}>"

    def to_dict(self, include_messages: bool = False) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_messages:
            data["messages"] = [m.to_dict() for m in self.messages.all()]
        return data


class Message(Base):
    """Individual message in a conversation."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(
        String(36), ForeignKey("conversations.id"), nullable=False, index=True
    )
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message {self.role.value}: {self.content[:30]}...>"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role.value,
            "content": self.content,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
