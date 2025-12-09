"""Conversation service for chat history management."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.conversation import Conversation, Message, MessageRole


class ConversationService:
    """Service for managing conversations and messages."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_conversation(
        self,
        agent_id: str,
        user_id: str | None = None,
        title: str | None = None,
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            agent_id=agent_id,
            user_id=user_id,
            title=title,
        )
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation

    async def get_conversation(
        self,
        conversation_id: str,
        user_id: str | None = None,
    ) -> Conversation | None:
        """Get a conversation by ID."""
        query = select(Conversation).where(Conversation.id == conversation_id)
        if user_id:
            query = query.where(Conversation.user_id == user_id)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_conversations(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        """List conversations with optional filters."""
        query = select(Conversation).order_by(Conversation.updated_at.desc())

        if agent_id:
            query = query.where(Conversation.agent_id == agent_id)
        if user_id:
            query = query.where(Conversation.user_id == user_id)

        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: str | None = None,
    ) -> bool:
        """Delete a conversation and its messages."""
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return False

        # Delete messages first
        messages_query = select(Message).where(
            Message.conversation_id == conversation_id
        )
        messages_result = await self.db.execute(messages_query)
        for message in messages_result.scalars().all():
            await self.db.delete(message)

        await self.db.delete(conversation)
        await self.db.commit()
        return True

    async def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        token_count: int | None = None,
    ) -> Message:
        """Add a message to a conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            token_count=token_count,
        )
        self.db.add(message)

        # Update conversation timestamp
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            from datetime import datetime, timezone
            conversation.updated_at = datetime.now(timezone.utc)

            # Auto-generate title from first user message
            if not conversation.title and role == MessageRole.USER:
                conversation.title = content[:100] + ("..." if len(content) > 100 else "")

        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def get_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        before_id: str | None = None,
    ) -> list[Message]:
        """Get messages for a conversation."""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )

        if before_id:
            # Get messages before a specific message (for pagination)
            before_msg = await self.db.execute(
                select(Message).where(Message.id == before_id)
            )
            before_msg = before_msg.scalar_one_or_none()
            if before_msg:
                query = query.where(Message.created_at < before_msg.created_at)

        query = query.limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_conversation_with_messages(
        self,
        conversation_id: str,
        user_id: str | None = None,
    ) -> Conversation | None:
        """Get conversation with all messages loaded."""
        query = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        if user_id:
            query = query.where(Conversation.user_id == user_id)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()
