"""
Unit tests for conversation memory management.
Tests various memory strategies and conversation history.
"""
import pytest
from datetime import datetime, timezone

from app.core.memory import (
    Message,
    MessageRole,
    ConversationHistory,
    MemoryStrategy,
    SlidingWindowMemory,
    TokenWindowMemory,
    SummaryMemory,
    BufferMemory,
    CompositeMemory,
    ConversationMemory,
    create_sliding_window_memory,
    create_token_limited_memory,
    create_summary_memory,
)


class TestMessageRole:
    """Tests for message role enum."""

    def test_role_values(self):
        """Test role enum values."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"

    def test_role_from_string(self):
        """Test creating role from string."""
        assert MessageRole("user") == MessageRole.USER
        assert MessageRole("assistant") == MessageRole.ASSISTANT


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_basic_message(self):
        """Test creating a basic message."""
        msg = Message(role=MessageRole.USER, content="Hello!")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.name is None

    def test_message_has_timestamp(self):
        """Test message has timestamp."""
        msg = Message(role=MessageRole.USER, content="Test")

        assert msg.timestamp is not None
        assert isinstance(msg.timestamp, datetime)

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = Message(role=MessageRole.USER, content="Hello!")
        result = msg.to_dict()

        assert result["role"] == "user"
        assert result["content"] == "Hello!"
        assert "timestamp" not in result  # Not included in API dict

    def test_message_with_name(self):
        """Test message with name field."""
        msg = Message(role=MessageRole.USER, content="Hi", name="John")
        result = msg.to_dict()

        assert result["name"] == "John"

    def test_message_with_tool_call_id(self):
        """Test message with tool call ID."""
        msg = Message(
            role=MessageRole.TOOL,
            content="Result",
            tool_call_id="call_123",
        )
        result = msg.to_dict()

        assert result["tool_call_id"] == "call_123"

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=tool_calls,
        )
        result = msg.to_dict()

        assert result["tool_calls"] == tool_calls


class TestConversationHistory:
    """Tests for conversation history."""

    def test_create_empty_history(self):
        """Test creating empty history."""
        history = ConversationHistory()

        assert len(history) == 0
        assert history.system_message is None

    def test_add_message(self):
        """Test adding messages."""
        history = ConversationHistory()
        msg = Message(role=MessageRole.USER, content="Hello")
        history.add_message(msg)

        assert len(history) == 1
        assert history.messages[0].content == "Hello"

    def test_add_user_message(self):
        """Test adding user message convenience method."""
        history = ConversationHistory()
        history.add_user_message("Hi there!")

        assert len(history) == 1
        assert history.messages[0].role == MessageRole.USER

    def test_add_assistant_message(self):
        """Test adding assistant message."""
        history = ConversationHistory()
        history.add_assistant_message("Hello!")

        assert len(history) == 1
        assert history.messages[0].role == MessageRole.ASSISTANT

    def test_set_system_message(self):
        """Test setting system message."""
        history = ConversationHistory()
        history.set_system_message("You are helpful.")

        assert history.system_message is not None
        assert history.system_message.content == "You are helpful."
        assert history.system_message.role == MessageRole.SYSTEM

    def test_get_messages_includes_system(self):
        """Test get_messages includes system message first."""
        history = ConversationHistory()
        history.set_system_message("System prompt")
        history.add_user_message("Hello")

        messages = history.get_messages()

        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.USER

    def test_to_list(self):
        """Test converting to list of dicts."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi!")

        result = history.to_list()

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_clear(self):
        """Test clearing history."""
        history = ConversationHistory()
        history.set_system_message("System")
        history.add_user_message("Hello")
        history.clear()

        assert len(history) == 0
        assert history.system_message is not None  # System preserved


class TestSlidingWindowMemory:
    """Tests for sliding window memory strategy."""

    def test_create_sliding_window(self):
        """Test creating sliding window memory."""
        memory = SlidingWindowMemory(window_size=5)
        assert memory.window_size == 5

    def test_invalid_window_size(self):
        """Test invalid window size raises error."""
        with pytest.raises(ValueError):
            SlidingWindowMemory(window_size=0)

    def test_keeps_recent_messages(self):
        """Test window keeps most recent messages."""
        memory = SlidingWindowMemory(window_size=3)
        history = ConversationHistory()

        for i in range(5):
            history.add_user_message(f"Message {i}")

        result = memory.apply(history)

        assert len(result) == 3
        assert result.messages[0].content == "Message 2"
        assert result.messages[2].content == "Message 4"

    def test_preserves_system_message(self):
        """Test system message is preserved."""
        memory = SlidingWindowMemory(window_size=2)
        history = ConversationHistory()
        history.set_system_message("System prompt")

        for i in range(5):
            history.add_user_message(f"Message {i}")

        result = memory.apply(history)

        assert result.system_message is not None
        assert result.system_message.content == "System prompt"

    def test_fewer_messages_than_window(self):
        """Test with fewer messages than window size."""
        memory = SlidingWindowMemory(window_size=10)
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi!")

        result = memory.apply(history)

        assert len(result) == 2


class TestTokenWindowMemory:
    """Tests for token window memory strategy."""

    def test_create_token_window(self):
        """Test creating token window memory."""
        memory = TokenWindowMemory(max_tokens=1000)
        assert memory.max_tokens == 1000

    def test_invalid_max_tokens(self):
        """Test invalid max tokens raises error."""
        with pytest.raises(ValueError):
            TokenWindowMemory(max_tokens=50)

    def test_fits_within_limit(self):
        """Test messages that fit within limit."""
        memory = TokenWindowMemory(max_tokens=1000)
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi!")

        result = memory.apply(history)

        assert len(result) == 2

    def test_truncates_old_messages(self):
        """Test old messages are truncated when exceeding token limit."""
        memory = TokenWindowMemory(max_tokens=150)  # Very small to force truncation
        history = ConversationHistory()

        # Add many messages with substantial content
        for i in range(10):
            history.add_user_message(
                f"This is message number {i} with lots of content that takes up many tokens. "
                f"The quick brown fox jumps over the lazy dog. Adding more text here."
            )

        result = memory.apply(history)

        # Should have fewer messages due to token limit
        assert len(result) < 10

    def test_preserves_recent_messages(self):
        """Test recent messages are kept."""
        memory = TokenWindowMemory(max_tokens=500)
        history = ConversationHistory()

        for i in range(10):
            history.add_user_message(f"Message {i}")

        result = memory.apply(history)

        # Last message should always be included if it fits
        if len(result) > 0:
            assert result.messages[-1].content == "Message 9"


class TestSummaryMemory:
    """Tests for summary memory strategy."""

    def test_create_summary_memory(self):
        """Test creating summary memory."""
        memory = SummaryMemory(summary_threshold=20, keep_recent=5)
        assert memory.summary_threshold == 20
        assert memory.keep_recent == 5

    def test_invalid_keep_recent(self):
        """Test invalid keep_recent raises error."""
        with pytest.raises(ValueError):
            SummaryMemory(summary_threshold=10, keep_recent=15)

    def test_no_summary_below_threshold(self):
        """Test no summarization when below threshold."""
        memory = SummaryMemory(summary_threshold=20, keep_recent=5)
        history = ConversationHistory()

        for i in range(10):
            history.add_user_message(f"Message {i}")

        result = memory.apply(history)

        # All messages preserved
        assert len(result) == 10

    def test_summarizes_when_above_threshold(self):
        """Test summarization when above threshold."""
        memory = SummaryMemory(summary_threshold=10, keep_recent=3)
        history = ConversationHistory()

        for i in range(15):
            history.add_user_message(f"Message {i}")

        result = memory.apply(history)

        # Should have summary + keep_recent messages
        assert len(result) == 4  # 1 summary + 3 recent

    def test_keeps_recent_messages(self):
        """Test recent messages are kept unchanged."""
        memory = SummaryMemory(summary_threshold=10, keep_recent=3)
        history = ConversationHistory()

        for i in range(15):
            history.add_user_message(f"Message {i}")

        result = memory.apply(history)

        # Last 3 messages should be kept
        recent = result.messages[-3:]
        assert recent[0].content == "Message 12"
        assert recent[1].content == "Message 13"
        assert recent[2].content == "Message 14"


class TestBufferMemory:
    """Tests for buffer memory strategy."""

    def test_buffer_returns_all(self):
        """Test buffer returns all messages."""
        memory = BufferMemory()
        history = ConversationHistory()

        for i in range(100):
            history.add_user_message(f"Message {i}")

        result = memory.apply(history)

        assert len(result) == 100


class TestCompositeMemory:
    """Tests for composite memory strategy."""

    def test_create_composite(self):
        """Test creating composite memory."""
        strategies = [
            SlidingWindowMemory(window_size=10),
            TokenWindowMemory(max_tokens=1000),
        ]
        memory = CompositeMemory(strategies)

        assert len(memory.strategies) == 2

    def test_empty_strategies_raises(self):
        """Test empty strategies raises error."""
        with pytest.raises(ValueError):
            CompositeMemory([])

    def test_applies_strategies_in_order(self):
        """Test strategies are applied in sequence."""
        strategies = [
            SlidingWindowMemory(window_size=10),  # First: keep last 10
            SlidingWindowMemory(window_size=5),  # Then: keep last 5
        ]
        memory = CompositeMemory(strategies)
        history = ConversationHistory()

        for i in range(20):
            history.add_user_message(f"Message {i}")

        result = memory.apply(history)

        assert len(result) == 5


class TestConversationMemory:
    """Tests for main ConversationMemory class."""

    def test_create_default_memory(self):
        """Test creating memory with defaults."""
        memory = ConversationMemory()

        assert memory.message_count == 0

    def test_create_with_system_prompt(self):
        """Test creating memory with system prompt."""
        memory = ConversationMemory(system_prompt="You are helpful.")

        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_add_message(self):
        """Test adding messages."""
        memory = ConversationMemory()
        memory.add_message(MessageRole.USER, "Hello")

        assert memory.message_count == 1

    def test_add_message_string_role(self):
        """Test adding message with string role."""
        memory = ConversationMemory()
        memory.add_message("user", "Hello")

        assert memory.message_count == 1

    def test_add_user_message(self):
        """Test add_user_message convenience method."""
        memory = ConversationMemory()
        memory.add_user_message("Hello!")

        messages = memory.get_raw_messages()
        assert messages[0].role == MessageRole.USER

    def test_add_assistant_message(self):
        """Test add_assistant_message convenience method."""
        memory = ConversationMemory()
        memory.add_assistant_message("Hi there!")

        messages = memory.get_raw_messages()
        assert messages[0].role == MessageRole.ASSISTANT

    def test_get_messages_applies_strategy(self):
        """Test get_messages applies memory strategy."""
        memory = ConversationMemory(
            strategy=SlidingWindowMemory(window_size=2),
        )

        for i in range(5):
            memory.add_user_message(f"Message {i}")

        messages = memory.get_messages()

        assert len(messages) == 2
        assert messages[0]["content"] == "Message 3"

    def test_clear(self):
        """Test clearing memory."""
        memory = ConversationMemory(system_prompt="System")
        memory.add_user_message("Hello")
        memory.clear()

        messages = memory.get_messages()
        assert len(messages) == 1  # Only system prompt

    def test_set_system_prompt(self):
        """Test setting system prompt."""
        memory = ConversationMemory()
        memory.set_system_prompt("New prompt")

        messages = memory.get_messages()
        assert messages[0]["content"] == "New prompt"

    def test_len(self):
        """Test len() returns message count."""
        memory = ConversationMemory()
        memory.add_user_message("1")
        memory.add_user_message("2")

        assert len(memory) == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_sliding_window_memory(self):
        """Test creating sliding window memory."""
        memory = create_sliding_window_memory(window_size=5)

        for i in range(10):
            memory.add_user_message(f"Message {i}")

        messages = memory.get_messages()
        assert len(messages) == 5

    def test_create_sliding_window_with_prompt(self):
        """Test creating sliding window with system prompt."""
        memory = create_sliding_window_memory(
            window_size=3,
            system_prompt="System",
        )

        for i in range(5):
            memory.add_user_message(f"Message {i}")

        messages = memory.get_messages()
        assert len(messages) == 4  # System + 3 messages
        assert messages[0]["role"] == "system"

    def test_create_token_limited_memory(self):
        """Test creating token limited memory."""
        memory = create_token_limited_memory(max_tokens=500)

        assert memory is not None
        assert isinstance(memory.strategy, TokenWindowMemory)

    def test_create_summary_memory(self):
        """Test creating summary memory."""
        memory = create_summary_memory(
            summary_threshold=15,
            keep_recent=3,
        )

        assert memory is not None
        assert isinstance(memory.strategy, SummaryMemory)
