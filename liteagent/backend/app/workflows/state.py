"""
Workflow state management.

Contains WorkflowDefinition and WorkflowState for managing workflow execution.
Implements Factor 5 (Unified execution and business state) and Factor 6 (Launch/Pause/Resume).
"""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.workflows.types import (
    Edge,
    NodeDefinition,
    NodeExecution,
    NodeStatus,
    NodeType,
    WorkflowStatus,
)


@dataclass
class WorkflowDefinition:
    """
    Declarative workflow definition.

    Defines the structure of a workflow without execution state.
    """
    id: str
    name: str
    nodes: list[NodeDefinition]
    edges: list[Edge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> NodeDefinition | None:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_outgoing_edges(self, node_id: str) -> list[Edge]:
        """Get edges leaving a node."""
        return [e for e in self.edges if e.source == node_id]

    def get_incoming_edges(self, node_id: str) -> list[Edge]:
        """Get edges entering a node."""
        return [e for e in self.edges if e.target == node_id]

    def get_start_node(self) -> NodeDefinition | None:
        """Get the START node."""
        for node in self.nodes:
            if node.type == NodeType.START:
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [{"source": e.source, "target": e.target, "condition": e.condition} for e in self.edges],
            "metadata": self.metadata,
        }


@dataclass
class WorkflowState:
    """
    Complete workflow execution state.

    Factor 5: Unified execution and business state.
    Factor 6: Enables pause/resume via serialization.
    Factor 12: Input to stateless reducer.
    """
    workflow_id: str
    execution_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_nodes: list[str] = field(default_factory=list)  # Active nodes
    node_executions: dict[str, NodeExecution] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)  # Workflow variables
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_node_execution(self, node_id: str) -> NodeExecution:
        """Get or create node execution record."""
        if node_id not in self.node_executions:
            self.node_executions[node_id] = NodeExecution(node_id=node_id)
        return self.node_executions[node_id]

    def set_variable(self, name: str, value: Any) -> None:
        """Set a workflow variable."""
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a workflow variable."""
        return self.variables.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "current_nodes": self.current_nodes,
            "node_executions": {k: v.to_dict() for k, v in self.node_executions.items()},
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowState":
        """Deserialize workflow state."""
        state = cls(
            workflow_id=data["workflow_id"],
            execution_id=data["execution_id"],
            status=WorkflowStatus(data["status"]),
            current_nodes=data.get("current_nodes", []),
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
        )

        for node_id, exec_data in data.get("node_executions", {}).items():
            state.node_executions[node_id] = NodeExecution(
                node_id=exec_data["node_id"],
                status=NodeStatus(exec_data["status"]),
                input_data=exec_data.get("input_data", {}),
                output_data=exec_data.get("output_data", {}),
                error=exec_data.get("error"),
            )

        return state

    def compute_hash(self) -> str:
        """Compute hash for state comparison."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
