"""
Workflow builder - fluent API for creating workflows.

Provides a declarative, chainable API for constructing workflow definitions.
"""
from typing import Any

from app.workflows.types import Edge, NodeDefinition, NodeType
from app.workflows.state import WorkflowDefinition


class WorkflowBuilder:
    """Fluent builder for workflow definitions."""

    def __init__(self, workflow_id: str, name: str):
        self.workflow_id = workflow_id
        self.name = name
        self._nodes: list[NodeDefinition] = []
        self._edges: list[Edge] = []
        self._metadata: dict[str, Any] = {}

    def add_start(self, node_id: str = "start") -> "WorkflowBuilder":
        """Add start node."""
        self._nodes.append(NodeDefinition(id=node_id, type=NodeType.START))
        return self

    def add_end(self, node_id: str = "end") -> "WorkflowBuilder":
        """Add end node."""
        self._nodes.append(NodeDefinition(id=node_id, type=NodeType.END))
        return self

    def add_agent(
        self,
        node_id: str,
        agent_id: str | None = None,
        purpose: str = "",
        input_variable: str = "input",
        output_variable: str = "output",
    ) -> "WorkflowBuilder":
        """Add agent node."""
        self._nodes.append(NodeDefinition(
            id=node_id,
            type=NodeType.AGENT,
            config={
                "agent_id": agent_id or node_id,
                "purpose": purpose,
                "input_variable": input_variable,
                "output_variable": output_variable,
            },
        ))
        return self

    def add_condition(
        self,
        node_id: str,
        condition: str,
    ) -> "WorkflowBuilder":
        """Add condition node."""
        self._nodes.append(NodeDefinition(
            id=node_id,
            type=NodeType.CONDITION,
            config={"condition": condition},
        ))
        return self

    def add_transform(
        self,
        node_id: str,
        transform: str = "passthrough",
        input_variable: str = "input",
        output_variable: str = "output",
        template: str | None = None,
    ) -> "WorkflowBuilder":
        """Add transform node."""
        config: dict[str, Any] = {
            "transform": transform,
            "input_variable": input_variable,
            "output_variable": output_variable,
        }
        if template:
            config["template"] = template
        self._nodes.append(NodeDefinition(
            id=node_id,
            type=NodeType.TRANSFORM,
            config=config,
        ))
        return self

    def add_human(
        self,
        node_id: str,
        prompt: str,
        options: list[str] | None = None,
    ) -> "WorkflowBuilder":
        """Add human checkpoint node."""
        self._nodes.append(NodeDefinition(
            id=node_id,
            type=NodeType.HUMAN,
            config={
                "prompt": prompt,
                "options": options or [],
            },
        ))
        return self

    def connect(
        self,
        source: str,
        target: str,
        condition: str | None = None,
    ) -> "WorkflowBuilder":
        """Add edge between nodes."""
        self._edges.append(Edge(source=source, target=target, condition=condition))
        return self

    def build(self) -> WorkflowDefinition:
        """Build the workflow definition."""
        return WorkflowDefinition(
            id=self.workflow_id,
            name=self.name,
            nodes=self._nodes,
            edges=self._edges,
            metadata=self._metadata,
        )
