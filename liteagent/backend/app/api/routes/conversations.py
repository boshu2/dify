"""Conversation routes for chat history."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.conversation import MessageRole
from app.services.conversation_service import ConversationService

router = APIRouter()


# Schemas
class ConversationCreate(BaseModel):
    """Create conversation request."""
    agent_id: str
    title: str | None = None


class MessageCreate(BaseModel):
    """Create message request."""
    role: str  # "user" or "assistant"
    content: str
    token_count: int | None = None


class MessageResponse(BaseModel):
    """Message response."""
    id: str
    conversation_id: str
    role: str
    content: str
    token_count: int | None
    created_at: str | None


class ConversationResponse(BaseModel):
    """Conversation response."""
    id: str
    agent_id: str
    user_id: str | None
    title: str | None
    created_at: str | None
    updated_at: str | None


class ConversationWithMessages(ConversationResponse):
    """Conversation with messages."""
    messages: list[MessageResponse]


@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    data: ConversationCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation."""
    service = ConversationService(db)
    conversation = await service.create_conversation(
        agent_id=data.agent_id,
        title=data.title,
    )
    return ConversationResponse(
        id=conversation.id,
        agent_id=conversation.agent_id,
        user_id=conversation.user_id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat() if conversation.created_at else None,
        updated_at=conversation.updated_at.isoformat() if conversation.updated_at else None,
    )


@router.get("/", response_model=list[ConversationResponse])
async def list_conversations(
    agent_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List conversations."""
    service = ConversationService(db)
    conversations = await service.list_conversations(
        agent_id=agent_id,
        limit=limit,
        offset=offset,
    )
    return [
        ConversationResponse(
            id=c.id,
            agent_id=c.agent_id,
            user_id=c.user_id,
            title=c.title,
            created_at=c.created_at.isoformat() if c.created_at else None,
            updated_at=c.updated_at.isoformat() if c.updated_at else None,
        )
        for c in conversations
    ]


@router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a conversation with messages."""
    service = ConversationService(db)
    conversation = await service.get_conversation_with_messages(conversation_id)

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages = await service.get_messages(conversation_id)

    return ConversationWithMessages(
        id=conversation.id,
        agent_id=conversation.agent_id,
        user_id=conversation.user_id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat() if conversation.created_at else None,
        updated_at=conversation.updated_at.isoformat() if conversation.updated_at else None,
        messages=[
            MessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role.value,
                content=m.content,
                token_count=m.token_count,
                created_at=m.created_at.isoformat() if m.created_at else None,
            )
            for m in messages
        ],
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation."""
    service = ConversationService(db)
    success = await service.delete_conversation(conversation_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return {"status": "deleted"}


@router.post("/{conversation_id}/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def add_message(
    conversation_id: str,
    data: MessageCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add a message to a conversation."""
    service = ConversationService(db)

    # Verify conversation exists
    conversation = await service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Map role string to enum
    try:
        role = MessageRole(data.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {data.role}. Must be 'user', 'assistant', or 'system'",
        )

    message = await service.add_message(
        conversation_id=conversation_id,
        role=role,
        content=data.content,
        token_count=data.token_count,
    )

    return MessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role.value,
        content=message.content,
        token_count=message.token_count,
        created_at=message.created_at.isoformat() if message.created_at else None,
    )


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    conversation_id: str,
    limit: int = 100,
    before_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Get messages for a conversation."""
    service = ConversationService(db)

    # Verify conversation exists
    conversation = await service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages = await service.get_messages(
        conversation_id=conversation_id,
        limit=limit,
        before_id=before_id,
    )

    return [
        MessageResponse(
            id=m.id,
            conversation_id=m.conversation_id,
            role=m.role.value,
            content=m.content,
            token_count=m.token_count,
            created_at=m.created_at.isoformat() if m.created_at else None,
        )
        for m in messages
    ]
