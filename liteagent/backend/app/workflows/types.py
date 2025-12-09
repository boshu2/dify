"""
Workflow type definitions.

Contains enums and basic data classes used throughout the workflow system.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Types of workflow nodes."""
    START = "start"
    END = "end"
    AGENT = "agent"  # Executes an agent
    CONDITION = "condition"  # Branching logic
    PARALLEL = "parallel"  # Parallel execution
    MERGE = "merge"  # Wait for parallel branches
    LOOP = "loop"  # Iteration
    HUMAN = "human"  # Human checkpoint
    TRANSFORM = "transform"  # Data transformation
    WAIT = "wait"  # Wait for external event
    HTTP_REQUEST = "http_request"  # HTTP API calls
    LLM = "llm"  # Direct LLM invocation
    CODE = "code"  # Code execution
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"  # RAG retrieval
    VARIABLE_AGGREGATOR = "variable_aggregator"  # Variable merging


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_HUMAN = "waiting_human"
    WAITING_EVENT = "waiting_event"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeStatus(str, Enum):
    """Individual node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Edge:
    """Connection between workflow nodes."""
    source: str
    target: str
    condition: str | None = None  # Optional condition expression


@dataclass
class NodeDefinition:
    """Definition of a workflow node."""
    id: str
    type: NodeType
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "config": self.config,
        }


@dataclass
class NodeExecution:
    """Execution record for a single node."""
    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
        }


@dataclass
class WorkflowEvent:
    """Event that triggers state transitions."""
    event_type: str
    node_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
