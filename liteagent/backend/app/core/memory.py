"""
Conversation memory management for LLM agents.
Provides strategies for managing conversation history within token limits.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Standard message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A conversation message."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to API-compatible dictionary."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result


@dataclass
class ConversationHistory:
    """Container for conversation messages."""

    messages: list[Message] = field(default_factory=list)
    system_message: Message | None = None
    conversation_id: str | None = None

    def add_message(self, message: Message) -> None:
        """Add a message to history."""
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message(Message(role=MessageRole.USER, content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.add_message(Message(role=MessageRole.ASSISTANT, content=content))

    def set_system_message(self, content: str) -> None:
        """Set the system message."""
        self.system_message = Message(role=MessageRole.SYSTEM, content=content)

    def get_messages(self) -> list[Message]:
        """Get all messages including system message."""
        if self.system_message:
            return [self.system_message] + self.messages
        return self.messages.copy()

    def to_list(self) -> list[dict[str, Any]]:
        """Convert to list of dictionaries for API calls."""
        return [m.to_dict() for m in self.get_messages()]

    def clear(self) -> None:
        """Clear conversation history (keeps system message)."""
        self.messages.clear()

    def __len__(self) -> int:
        """Return number of messages (excluding system)."""
        return len(self.messages)


class MemoryStrategy(ABC):
    """Base class for memory management strategies."""

    @abstractmethod
    def apply(self, history: ConversationHistory) -> ConversationHistory:
        """
        Apply memory strategy to conversation history.

        Args:
            history: The conversation history to process.

        Returns:
            Processed conversation history.
        """
        pass


class SlidingWindowMemory(MemoryStrategy):
    """
    Keep only the last N messages (sliding window).
    Always preserves system message.
    """

    def __init__(self, window_size: int = 10):
        """
        Initialize sliding window memory.

        Args:
            window_size: Maximum number of messages to keep.
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        self.window_size = window_size

    def apply(self, history: ConversationHistory) -> ConversationHistory:
        """Keep only the last window_size messages."""
        result = ConversationHistory(
            system_message=history.system_message,
            conversation_id=history.conversation_id,
        )

        # Keep only the last window_size messages
        result.messages = history.messages[-self.window_size :]

        return result


class TokenWindowMemory(MemoryStrategy):
    """
    Keep messages that fit within a token limit.
    Removes oldest messages first when over limit.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        token_counter: Any | None = None,
    ):
        """
        Initialize token window memory.

        Args:
            max_tokens: Maximum tokens allowed.
            token_counter: Optional token counter (uses estimate if None).
        """
        if max_tokens < 100:
            raise ValueError("Max tokens must be at least 100")
        self.max_tokens = max_tokens
        self.token_counter = token_counter

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self.token_counter:
            return self.token_counter.count(text)
        # Rough estimate: ~4 chars per token
        return len(text) // 4

    def _message_tokens(self, message: Message) -> int:
        """Estimate tokens for a message."""
        # Base overhead for message format
        overhead = 4
        content_tokens = self._estimate_tokens(message.content)
        return overhead + content_tokens

    def apply(self, history: ConversationHistory) -> ConversationHistory:
        """Keep messages that fit within token limit."""
        result = ConversationHistory(
            system_message=history.system_message,
            conversation_id=history.conversation_id,
        )

        # Calculate system message tokens
        system_tokens = 0
        if history.system_message:
            system_tokens = self._message_tokens(history.system_message)

        remaining_tokens = self.max_tokens - system_tokens
        included_messages: list[Message] = []

        # Start from most recent, add until we hit limit
        for message in reversed(history.messages):
            msg_tokens = self._message_tokens(message)
            if msg_tokens <= remaining_tokens:
                included_messages.insert(0, message)
                remaining_tokens -= msg_tokens
            else:
                # Can't fit more messages
                break

        result.messages = included_messages
        return result


class SummaryMemory(MemoryStrategy):
    """
    Summarize older messages when history gets too long.
    Keeps recent messages and a summary of older ones.
    """

    def __init__(
        self,
        summary_threshold: int = 20,
        keep_recent: int = 5,
        summarizer: Any | None = None,
    ):
        """
        Initialize summary memory.

        Args:
            summary_threshold: Trigger summarization when messages exceed this.
            keep_recent: Number of recent messages to always keep.
            summarizer: Optional async function to generate summaries.
        """
        if keep_recent >= summary_threshold:
            raise ValueError("keep_recent must be less than summary_threshold")
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent
        self.summarizer = summarizer
        self._cached_summary: str | None = None

    def _generate_summary(self, messages: list[Message]) -> str:
        """Generate a summary of messages (placeholder)."""
        if self.summarizer:
            # Would call async summarizer
            pass

        # Simple placeholder summary
        user_msgs = [m for m in messages if m.role == MessageRole.USER]
        assistant_msgs = [m for m in messages if m.role == MessageRole.ASSISTANT]

        return (
            f"[Summary of {len(messages)} messages: "
            f"{len(user_msgs)} user, {len(assistant_msgs)} assistant messages]"
        )

    def apply(self, history: ConversationHistory) -> ConversationHistory:
        """Summarize older messages if over threshold."""
        result = ConversationHistory(
            system_message=history.system_message,
            conversation_id=history.conversation_id,
        )

        if len(history.messages) <= self.summary_threshold:
            # No need to summarize
            result.messages = history.messages.copy()
            return result

        # Split into old and recent
        split_point = len(history.messages) - self.keep_recent
        old_messages = history.messages[:split_point]
        recent_messages = history.messages[split_point:]

        # Generate summary of old messages
        summary = self._generate_summary(old_messages)

        # Add summary as assistant message at start
        summary_message = Message(
            role=MessageRole.ASSISTANT,
            content=summary,
            metadata={"is_summary": True},
        )

        result.messages = [summary_message] + recent_messages
        return result


class BufferMemory(MemoryStrategy):
    """
    Simple buffer that keeps all messages.
    No truncation or summarization.
    """

    def apply(self, history: ConversationHistory) -> ConversationHistory:
        """Return history unchanged."""
        return ConversationHistory(
            messages=history.messages.copy(),
            system_message=history.system_message,
            conversation_id=history.conversation_id,
        )


class CompositeMemory(MemoryStrategy):
    """
    Apply multiple memory strategies in sequence.
    Useful for combining token limits with sliding window.
    """

    def __init__(self, strategies: list[MemoryStrategy]):
        """
        Initialize composite memory.

        Args:
            strategies: List of strategies to apply in order.
        """
        if not strategies:
            raise ValueError("At least one strategy is required")
        self.strategies = strategies

    def apply(self, history: ConversationHistory) -> ConversationHistory:
        """Apply all strategies in sequence."""
        result = history
        for strategy in self.strategies:
            result = strategy.apply(result)
        return result


class ConversationMemory:
    """
    Main class for managing conversation memory.
    Combines history storage with memory strategy.
    """

    def __init__(
        self,
        strategy: MemoryStrategy | None = None,
        system_prompt: str | None = None,
        conversation_id: str | None = None,
    ):
        """
        Initialize conversation memory.

        Args:
            strategy: Memory management strategy.
            system_prompt: Optional system prompt.
            conversation_id: Optional conversation identifier.
        """
        self.strategy = strategy or BufferMemory()
        self.history = ConversationHistory(conversation_id=conversation_id)

        if system_prompt:
            self.history.set_system_message(system_prompt)

    def add_message(self, role: MessageRole | str, content: str, **kwargs) -> None:
        """
        Add a message to memory.

        Args:
            role: Message role.
            content: Message content.
            **kwargs: Additional message fields.
        """
        if isinstance(role, str):
            role = MessageRole(role)

        message = Message(role=role, content=content, **kwargs)
        self.history.add_message(message)

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message(MessageRole.USER, content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.add_message(MessageRole.ASSISTANT, content)

    def get_messages(self) -> list[dict[str, Any]]:
        """Get messages after applying memory strategy."""
        processed = self.strategy.apply(self.history)
        return processed.to_list()

    def get_raw_messages(self) -> list[Message]:
        """Get raw Message objects."""
        return self.history.get_messages()

    def clear(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """Set or update system prompt."""
        self.history.set_system_message(prompt)

    @property
    def message_count(self) -> int:
        """Get number of messages (excluding system)."""
        return len(self.history)

    def __len__(self) -> int:
        """Return message count."""
        return self.message_count


# Factory functions for common configurations
def create_sliding_window_memory(
    window_size: int = 10,
    system_prompt: str | None = None,
) -> ConversationMemory:
    """Create memory with sliding window strategy."""
    return ConversationMemory(
        strategy=SlidingWindowMemory(window_size=window_size),
        system_prompt=system_prompt,
    )


def create_token_limited_memory(
    max_tokens: int = 4000,
    system_prompt: str | None = None,
) -> ConversationMemory:
    """Create memory with token limit strategy."""
    return ConversationMemory(
        strategy=TokenWindowMemory(max_tokens=max_tokens),
        system_prompt=system_prompt,
    )


def create_summary_memory(
    summary_threshold: int = 20,
    keep_recent: int = 5,
    system_prompt: str | None = None,
) -> ConversationMemory:
    """Create memory with summary strategy."""
    return ConversationMemory(
        strategy=SummaryMemory(
            summary_threshold=summary_threshold,
            keep_recent=keep_recent,
        ),
        system_prompt=system_prompt,
    )
