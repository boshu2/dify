"""Tests for conversation service."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.models.conversation import Conversation, Message, MessageRole
from app.services.conversation_service import ConversationService


class TestConversationService:
    """Tests for ConversationService."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        db.execute = AsyncMock()
        db.delete = AsyncMock()
        return db

    @pytest.fixture
    def service(self, mock_db):
        return ConversationService(mock_db)

    @pytest.mark.asyncio
    async def test_create_conversation(self, service, mock_db):
        """Test creating a conversation."""
        conversation = await service.create_conversation(
            agent_id="agent-1",
            user_id="user-1",
            title="Test Conversation",
        )

        assert conversation.agent_id == "agent-1"
        assert conversation.user_id == "user-1"
        assert conversation.title == "Test Conversation"
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_get_conversation(self, service, mock_db):
        """Test getting a conversation."""
        conv = Conversation(
            id="conv-1",
            agent_id="agent-1",
            user_id="user-1",
            title="Test",
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = conv
        mock_db.execute.return_value = mock_result

        result = await service.get_conversation("conv-1")

        assert result.id == "conv-1"

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, service, mock_db):
        """Test getting non-existent conversation."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await service.get_conversation("not-found")

        assert result is None

    @pytest.mark.asyncio
    async def test_add_message(self, service, mock_db):
        """Test adding a message."""
        # Mock get_conversation
        conv = Conversation(id="conv-1", agent_id="agent-1")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = conv
        mock_db.execute.return_value = mock_result

        message = await service.add_message(
            conversation_id="conv-1",
            role=MessageRole.USER,
            content="Hello, world!",
            token_count=3,
        )

        assert message.conversation_id == "conv-1"
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.token_count == 3


class TestConversationModel:
    """Tests for Conversation model."""

    def test_conversation_to_dict(self):
        """Test conversation to_dict method."""
        from datetime import datetime, timezone

        conv = Conversation(
            id="conv-1",
            agent_id="agent-1",
            user_id="user-1",
            title="Test",
        )
        conv.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        conv.updated_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        data = conv.to_dict()

        assert data["id"] == "conv-1"
        assert data["agent_id"] == "agent-1"
        assert data["title"] == "Test"


class TestMessageModel:
    """Tests for Message model."""

    def test_message_to_dict(self):
        """Test message to_dict method."""
        from datetime import datetime, timezone

        msg = Message(
            id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.USER,
            content="Hello",
            token_count=1,
        )
        msg.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        data = msg.to_dict()

        assert data["id"] == "msg-1"
        assert data["role"] == "user"
        assert data["content"] == "Hello"


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_message_role_values(self):
        """Test message role enum values."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"
