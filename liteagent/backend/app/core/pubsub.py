"""
Redis Pub/Sub for real-time agent state updates.

Enables:
- Real-time streaming of agent execution events
- Multi-agent coordination
- Distributed workflow control signals
"""
import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generator, Iterator

from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Agent event types for pub/sub."""

    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_STEP = "agent.step"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_PAUSED = "agent.paused"

    # Streaming events
    STREAM_CHUNK = "stream.chunk"
    STREAM_TOOL_CALL = "stream.tool_call"
    STREAM_TOOL_RESULT = "stream.tool_result"
    STREAM_END = "stream.end"

    # Control events
    CONTROL_PAUSE = "control.pause"
    CONTROL_RESUME = "control.resume"
    CONTROL_ABORT = "control.abort"

    # Human-in-the-loop
    HUMAN_INPUT_REQUESTED = "human.input_requested"
    HUMAN_INPUT_PROVIDED = "human.input_provided"


@dataclass
class AgentEvent:
    """Event published to agent channels."""

    event_type: EventType
    agent_id: str
    execution_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "execution_id": self.execution_id,
            "data": self.data,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, data: str | bytes) -> "AgentEvent":
        """Deserialize from JSON."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        parsed = json.loads(data)
        return cls(
            event_type=EventType(parsed["event_type"]),
            agent_id=parsed["agent_id"],
            execution_id=parsed["execution_id"],
            data=parsed.get("data", {}),
            timestamp=parsed.get("timestamp", ""),
        )


class Topic:
    """
    Pub/Sub topic for agent events.

    Supports both publishing and subscribing to events.
    """

    KEY_PREFIX = "liteagent:pubsub"

    def __init__(self, topic_name: str):
        self.topic_name = topic_name
        self._key = f"{self.KEY_PREFIX}:{topic_name}"

    def publish(self, event: AgentEvent) -> int:
        """
        Publish event to topic.

        Returns:
            Number of subscribers that received the message.
        """
        try:
            return redis_client.publish(self._key, event.to_json())
        except Exception as e:
            logger.warning(f"Publish failed: {e}")
            return 0

    def subscribe(self) -> "Subscription":
        """Create a subscription to this topic."""
        return Subscription(self._key)


class Subscription:
    """
    Thread-safe subscription to a pub/sub topic.

    Features:
    - Background listener thread
    - Bounded message queue with backpressure
    - Graceful shutdown
    """

    def __init__(
        self,
        channel: str,
        max_queue_size: int = 1024,
    ):
        self._channel = channel
        self._max_queue_size = max_queue_size
        self._queue: queue.Queue[AgentEvent | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._pubsub = None
        self._listener_thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._started = False
        self._start_lock = threading.Lock()
        self._dropped_count = 0

    def _start_if_needed(self) -> None:
        """Lazy start subscription on first access."""
        with self._start_lock:
            if self._started:
                return

            self._pubsub = redis_client.pubsub()
            self._pubsub.subscribe(self._channel)

            self._listener_thread = threading.Thread(
                target=self._listen,
                name=f"pubsub-{self._channel}",
                daemon=True,
            )
            self._listener_thread.start()
            self._started = True

    def _listen(self) -> None:
        """Main listener loop in background thread."""
        while not self._closed.is_set():
            try:
                message = self._pubsub.get_message(timeout=1.0)

                if message is None:
                    continue

                if message.get("type") != "message":
                    continue

                payload = message.get("data")
                if isinstance(payload, bytes):
                    event = AgentEvent.from_json(payload)
                    self._enqueue(event)

            except Exception as e:
                if not self._closed.is_set():
                    logger.error(f"Subscription error: {e}")
                break

    def _enqueue(self, event: AgentEvent) -> None:
        """Add event to queue, dropping oldest if full."""
        while not self._closed.is_set():
            try:
                self._queue.put_nowait(event)
                return
            except queue.Full:
                # Drop oldest message
                try:
                    self._queue.get_nowait()
                    self._dropped_count += 1
                except queue.Empty:
                    continue

    def receive(self, timeout: float | None = None) -> AgentEvent | None:
        """
        Receive next event from subscription.

        Args:
            timeout: Seconds to wait (None = block forever)

        Returns:
            AgentEvent or None if timeout
        """
        self._start_if_needed()

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __iter__(self) -> Iterator[AgentEvent]:
        """Iterate over events until closed."""
        self._start_if_needed()

        while not self._closed.is_set():
            event = self.receive(timeout=1.0)
            if event is not None:
                yield event

    def close(self) -> None:
        """Close subscription and cleanup resources."""
        self._closed.set()

        if self._pubsub:
            try:
                self._pubsub.unsubscribe(self._channel)
                self._pubsub.close()
            except Exception:
                pass

        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class AgentEventPublisher:
    """
    Publisher for agent execution events.

    Provides convenient methods for common event types.
    """

    def __init__(self, agent_id: str, execution_id: str):
        self.agent_id = agent_id
        self.execution_id = execution_id
        self._topic = Topic(f"agent:{agent_id}:{execution_id}")
        self._global_topic = Topic(f"agent:{agent_id}")

    def _publish(self, event_type: EventType, data: dict[str, Any] = None) -> None:
        """Publish event to both execution and agent topics."""
        event = AgentEvent(
            event_type=event_type,
            agent_id=self.agent_id,
            execution_id=self.execution_id,
            data=data or {},
        )
        self._topic.publish(event)
        self._global_topic.publish(event)

    def started(self, input_message: str) -> None:
        """Publish agent started event."""
        self._publish(EventType.AGENT_STARTED, {"input": input_message})

    def step(self, step_number: int, step_type: str, content: str) -> None:
        """Publish agent step event."""
        self._publish(EventType.AGENT_STEP, {
            "step_number": step_number,
            "step_type": step_type,
            "content": content,
        })

    def completed(self, result: str, total_steps: int) -> None:
        """Publish agent completed event."""
        self._publish(EventType.AGENT_COMPLETED, {
            "result": result,
            "total_steps": total_steps,
        })

    def failed(self, error: str) -> None:
        """Publish agent failed event."""
        self._publish(EventType.AGENT_FAILED, {"error": error})

    def paused(self, reason: str) -> None:
        """Publish agent paused event."""
        self._publish(EventType.AGENT_PAUSED, {"reason": reason})

    def stream_chunk(self, content: str, chunk_index: int) -> None:
        """Publish streaming chunk."""
        self._publish(EventType.STREAM_CHUNK, {
            "content": content,
            "chunk_index": chunk_index,
        })

    def stream_tool_call(
        self,
        tool_name: str,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> None:
        """Publish tool call event."""
        self._publish(EventType.STREAM_TOOL_CALL, {
            "tool_name": tool_name,
            "tool_id": tool_id,
            "arguments": arguments,
        })

    def stream_tool_result(
        self,
        tool_id: str,
        result: Any,
    ) -> None:
        """Publish tool result event."""
        self._publish(EventType.STREAM_TOOL_RESULT, {
            "tool_id": tool_id,
            "result": result,
        })

    def stream_end(self) -> None:
        """Publish stream end event."""
        self._publish(EventType.STREAM_END)

    def human_input_requested(self, question: str, context: dict = None) -> None:
        """Publish human input request."""
        self._publish(EventType.HUMAN_INPUT_REQUESTED, {
            "question": question,
            "context": context or {},
        })


class AgentEventSubscriber:
    """
    Subscriber for agent execution events.

    Provides filtered access to specific event types.
    """

    def __init__(self, agent_id: str, execution_id: str | None = None):
        self.agent_id = agent_id
        self.execution_id = execution_id

        if execution_id:
            self._topic = Topic(f"agent:{agent_id}:{execution_id}")
        else:
            self._topic = Topic(f"agent:{agent_id}")

    def subscribe(
        self,
        event_types: list[EventType] | None = None,
    ) -> Generator[AgentEvent, None, None]:
        """
        Subscribe to agent events.

        Args:
            event_types: Filter to specific event types (None = all)

        Yields:
            AgentEvent instances
        """
        with self._topic.subscribe() as subscription:
            for event in subscription:
                if event_types is None or event.event_type in event_types:
                    yield event

    def subscribe_streaming(self) -> Generator[AgentEvent, None, None]:
        """Subscribe to streaming events only."""
        return self.subscribe([
            EventType.STREAM_CHUNK,
            EventType.STREAM_TOOL_CALL,
            EventType.STREAM_TOOL_RESULT,
            EventType.STREAM_END,
        ])


class CommandChannel:
    """
    Redis-based command channel for agent control.

    Allows sending control signals (pause, resume, abort) to running agents.
    Uses Redis lists for reliable delivery.
    """

    KEY_PREFIX = "liteagent:commands"
    COMMAND_TTL = 3600  # 1 hour

    def __init__(self, agent_id: str, execution_id: str):
        self.agent_id = agent_id
        self.execution_id = execution_id
        self._key = f"{self.KEY_PREFIX}:{agent_id}:{execution_id}"
        self._pending_key = f"{self._key}:pending"

    def send_command(
        self,
        command_type: str,
        data: dict[str, Any] = None,
    ) -> None:
        """
        Send command to running agent.

        Args:
            command_type: Type of command (pause, resume, abort)
            data: Additional command data
        """
        command = {
            "command_type": command_type,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            with redis_client.pipeline() as pipe:
                pipe.rpush(self._key, json.dumps(command))
                pipe.expire(self._key, self.COMMAND_TTL)
                pipe.set(self._pending_key, "1", ex=self.COMMAND_TTL)
                pipe.execute()
        except Exception as e:
            logger.error(f"Failed to send command: {e}")

    def fetch_commands(self) -> list[dict[str, Any]]:
        """
        Fetch and clear all pending commands.

        Returns:
            List of command dictionaries
        """
        try:
            # Check if there are pending commands
            pending = redis_client.get(self._pending_key)
            if not pending:
                return []

            # Atomically get and delete commands
            with redis_client.pipeline() as pipe:
                pipe.lrange(self._key, 0, -1)
                pipe.delete(self._key)
                pipe.delete(self._pending_key)
                results = pipe.execute()

            commands = []
            for cmd_bytes in results[0] or []:
                try:
                    cmd = json.loads(cmd_bytes.decode("utf-8"))
                    commands.append(cmd)
                except (json.JSONDecodeError, AttributeError):
                    continue

            return commands

        except Exception as e:
            logger.error(f"Failed to fetch commands: {e}")
            return []

    def pause(self, reason: str = "User requested") -> None:
        """Send pause command."""
        self.send_command("pause", {"reason": reason})

    def resume(self) -> None:
        """Send resume command."""
        self.send_command("resume")

    def abort(self, reason: str = "User requested") -> None:
        """Send abort command."""
        self.send_command("abort", {"reason": reason})

    def has_pending_commands(self) -> bool:
        """Check if there are pending commands."""
        try:
            return bool(redis_client.exists(self._pending_key))
        except Exception:
            return False
