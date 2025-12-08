"""
Workflow Engine with Stateless Reducers.

Implements a workflow orchestration system following 12-factor principles:
- Factor 5: Unified execution and business state
- Factor 6: Launch/Pause/Resume
- Factor 8: Own your control flow
- Factor 12: Stateless reducer pattern

Workflows are defined declaratively and executed via pure state transformations.
"""
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
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


@dataclass
class WorkflowEvent:
    """Event that triggers state transitions."""
    event_type: str
    node_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NodeHandler(ABC):
    """Abstract handler for executing workflow nodes."""

    @abstractmethod
    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a node and return output data.

        Args:
            node: Node definition.
            state: Current workflow state.
            context: Execution context with dependencies.

        Returns:
            Output data from node execution.
        """
        pass


class StartNodeHandler(NodeHandler):
    """Handler for START nodes."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # START node just passes through input
        return state.variables.copy()


class EndNodeHandler(NodeHandler):
    """Handler for END nodes."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # END node collects final output
        return {"final_output": state.variables}


class AgentNodeHandler(NodeHandler):
    """Handler for AGENT nodes that run AI agents."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        from app.agents.twelve_factor_agent import Agent, AgentConfig, AgentStatus

        # Get agent from context or create from config
        agent = context.get("agents", {}).get(node.config.get("agent_id"))

        if not agent:
            # Create agent from config
            config = AgentConfig(
                agent_id=node.config.get("agent_id", node.id),
                purpose=node.config.get("purpose", "Execute task"),
                llm_client=context.get("llm_client"),
            )
            agent = Agent(config)

        # Get input message from variables or config
        input_var = node.config.get("input_variable", "input")
        user_message = state.get_variable(input_var, node.config.get("default_message", ""))

        # Run agent
        agent_state = agent.launch(str(user_message))
        agent_state = await agent.run_to_completion(agent_state)

        # Extract result
        result = ""
        if agent_state.status == AgentStatus.COMPLETED:
            # Get last assistant message
            for step in reversed(agent_state.steps):
                if step.step_type.value == "assistant_message":
                    result = step.content
                    break

        output_var = node.config.get("output_variable", "output")
        return {output_var: result, "agent_state": agent_state.to_dict()}


class ConditionNodeHandler(NodeHandler):
    """Handler for CONDITION nodes (branching)."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # Evaluate condition expression
        condition = node.config.get("condition", "True")

        # Simple expression evaluation with workflow variables
        try:
            # Create safe evaluation context
            eval_context = {"variables": state.variables, **state.variables}
            result = eval(condition, {"__builtins__": {}}, eval_context)
            return {"condition_result": bool(result), "branch": "true" if result else "false"}
        except Exception as e:
            return {"condition_result": False, "branch": "false", "error": str(e)}


class TransformNodeHandler(NodeHandler):
    """Handler for TRANSFORM nodes (data transformation)."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        transform_type = node.config.get("transform", "passthrough")
        input_var = node.config.get("input_variable", "input")
        output_var = node.config.get("output_variable", "output")

        input_data = state.get_variable(input_var)

        if transform_type == "passthrough":
            output_data = input_data
        elif transform_type == "json_parse":
            output_data = json.loads(input_data) if isinstance(input_data, str) else input_data
        elif transform_type == "json_stringify":
            output_data = json.dumps(input_data)
        elif transform_type == "template":
            template = node.config.get("template", "{input}")
            output_data = template.format(**state.variables)
        else:
            output_data = input_data

        return {output_var: output_data}


class HumanNodeHandler(NodeHandler):
    """Handler for HUMAN nodes (human-in-the-loop checkpoints)."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # Human nodes pause for input
        return {
            "waiting_for": "human_input",
            "prompt": node.config.get("prompt", "Please provide input"),
            "options": node.config.get("options", []),
        }


class WorkflowReducer:
    """
    Stateless workflow reducer.

    Factor 12: Pure function that transforms state based on events.
    Factor 8: Owns control flow through explicit state transitions.
    """

    def __init__(self):
        self._handlers: dict[NodeType, NodeHandler] = {
            NodeType.START: StartNodeHandler(),
            NodeType.END: EndNodeHandler(),
            NodeType.AGENT: AgentNodeHandler(),
            NodeType.CONDITION: ConditionNodeHandler(),
            NodeType.TRANSFORM: TransformNodeHandler(),
            NodeType.HUMAN: HumanNodeHandler(),
        }

    def register_handler(self, node_type: NodeType, handler: NodeHandler) -> None:
        """Register a custom node handler."""
        self._handlers[node_type] = handler

    async def reduce(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        event: WorkflowEvent,
        context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """
        Apply event to state and return new state.

        This is a pure function - same inputs always produce same outputs.
        """
        context = context or {}
        state.updated_at = datetime.now(timezone.utc)

        if event.event_type == "start":
            return self._handle_start(definition, state, event)

        elif event.event_type == "node_complete":
            return await self._handle_node_complete(definition, state, event, context)

        elif event.event_type == "node_failed":
            return self._handle_node_failed(definition, state, event)

        elif event.event_type == "human_response":
            return self._handle_human_response(definition, state, event)

        elif event.event_type == "pause":
            return self._handle_pause(state)

        elif event.event_type == "resume":
            return self._handle_resume(state)

        return state

    def _handle_start(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        event: WorkflowEvent,
    ) -> WorkflowState:
        """Handle workflow start event."""
        start_node = definition.get_start_node()
        if not start_node:
            state.status = WorkflowStatus.FAILED
            return state

        state.status = WorkflowStatus.RUNNING
        state.current_nodes = [start_node.id]

        # Set initial variables from event data
        for key, value in event.data.items():
            state.set_variable(key, value)

        # Mark start node as running
        exec_record = state.get_node_execution(start_node.id)
        exec_record.status = NodeStatus.RUNNING
        exec_record.started_at = datetime.now(timezone.utc)
        exec_record.input_data = event.data

        return state

    async def _handle_node_complete(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        event: WorkflowEvent,
        context: dict[str, Any],
    ) -> WorkflowState:
        """Handle node completion event."""
        node_id = event.node_id
        if not node_id:
            return state

        # Update node execution record
        exec_record = state.get_node_execution(node_id)
        exec_record.status = NodeStatus.COMPLETED
        exec_record.completed_at = datetime.now(timezone.utc)
        exec_record.output_data = event.data

        # Update workflow variables from output
        for key, value in event.data.items():
            state.set_variable(key, value)

        # Remove from current nodes
        if node_id in state.current_nodes:
            state.current_nodes.remove(node_id)

        # Get node definition
        node = definition.get_node(node_id)
        if not node:
            return state

        # Check if this is an END node
        if node.type == NodeType.END:
            if not state.current_nodes:
                state.status = WorkflowStatus.COMPLETED
            return state

        # Find next nodes
        next_nodes = self._get_next_nodes(definition, state, node_id, event.data)

        # Start next nodes
        for next_node_id in next_nodes:
            next_node = definition.get_node(next_node_id)
            if next_node:
                state.current_nodes.append(next_node_id)
                exec_record = state.get_node_execution(next_node_id)
                exec_record.status = NodeStatus.RUNNING
                exec_record.started_at = datetime.now(timezone.utc)

                # Check for human nodes
                if next_node.type == NodeType.HUMAN:
                    state.status = WorkflowStatus.WAITING_HUMAN

        # Check if workflow is complete
        if not state.current_nodes:
            state.status = WorkflowStatus.COMPLETED

        return state

    def _handle_node_failed(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        event: WorkflowEvent,
    ) -> WorkflowState:
        """Handle node failure event."""
        node_id = event.node_id
        if not node_id:
            return state

        exec_record = state.get_node_execution(node_id)
        exec_record.status = NodeStatus.FAILED
        exec_record.completed_at = datetime.now(timezone.utc)
        exec_record.error = event.data.get("error", "Unknown error")

        # Remove from current nodes
        if node_id in state.current_nodes:
            state.current_nodes.remove(node_id)

        # Mark workflow as failed
        state.status = WorkflowStatus.FAILED

        return state

    def _handle_human_response(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        event: WorkflowEvent,
    ) -> WorkflowState:
        """Handle human response event."""
        node_id = event.node_id
        if not node_id:
            return state

        # Store human response
        response_var = event.data.get("variable", "human_response")
        state.set_variable(response_var, event.data.get("response"))

        # Complete the human node
        exec_record = state.get_node_execution(node_id)
        exec_record.status = NodeStatus.COMPLETED
        exec_record.completed_at = datetime.now(timezone.utc)
        exec_record.output_data = event.data

        # Remove from current nodes
        if node_id in state.current_nodes:
            state.current_nodes.remove(node_id)

        # Find next nodes and add them
        next_nodes = self._get_next_nodes(definition, state, node_id, event.data)
        for next_node_id in next_nodes:
            if next_node_id not in state.current_nodes:
                state.current_nodes.append(next_node_id)
                next_exec = state.get_node_execution(next_node_id)
                next_exec.status = NodeStatus.RUNNING
                next_exec.started_at = datetime.now(timezone.utc)

        # Resume workflow
        state.status = WorkflowStatus.RUNNING

        return state

    def _handle_pause(self, state: WorkflowState) -> WorkflowState:
        """Handle pause event."""
        if state.status == WorkflowStatus.RUNNING:
            state.status = WorkflowStatus.PAUSED
        return state

    def _handle_resume(self, state: WorkflowState) -> WorkflowState:
        """Handle resume event."""
        if state.status == WorkflowStatus.PAUSED:
            state.status = WorkflowStatus.RUNNING
        return state

    def _get_next_nodes(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        node_id: str,
        output_data: dict[str, Any],
    ) -> list[str]:
        """Determine next nodes based on edges and conditions."""
        next_nodes = []
        outgoing = definition.get_outgoing_edges(node_id)

        for edge in outgoing:
            if edge.condition:
                # Evaluate edge condition
                try:
                    eval_context = {
                        "variables": state.variables,
                        "output": output_data,
                        **state.variables,
                        **output_data,
                    }
                    if eval(edge.condition, {"__builtins__": {}}, eval_context):
                        next_nodes.append(edge.target)
                except Exception:
                    pass  # Skip edge if condition fails
            else:
                next_nodes.append(edge.target)

        return next_nodes

    async def execute_node(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        node_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single node and return its output."""
        node = definition.get_node(node_id)
        if not node:
            raise ValueError(f"Node not found: {node_id}")

        handler = self._handlers.get(node.type)
        if not handler:
            raise ValueError(f"No handler for node type: {node.type}")

        return await handler.execute(node, state, context)


class WorkflowEngine:
    """
    Workflow execution engine.

    Orchestrates workflow execution using the reducer pattern.
    """

    def __init__(self):
        self.reducer = WorkflowReducer()

    def launch(
        self,
        definition: WorkflowDefinition,
        input_data: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """
        Launch a new workflow execution.

        Factor 6: Launch capability.
        Factor 11: Entry point for triggering workflow.
        """
        state = WorkflowState(
            workflow_id=definition.id,
            execution_id=str(uuid.uuid4()),
        )
        return state

    async def start(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        input_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """Start workflow execution."""
        event = WorkflowEvent(
            event_type="start",
            data=input_data or {},
        )
        state = await self.reducer.reduce(definition, state, event, context)

        # Execute start node
        if state.current_nodes:
            state = await self._execute_current_nodes(definition, state, context or {})

        return state

    async def step(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """
        Execute one step of the workflow.

        Factor 12: Stateless step execution.
        """
        if state.status not in [WorkflowStatus.RUNNING]:
            return state

        return await self._execute_current_nodes(definition, state, context or {})

    async def _execute_current_nodes(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> WorkflowState:
        """Execute all current nodes."""
        nodes_to_execute = list(state.current_nodes)

        for node_id in nodes_to_execute:
            node = definition.get_node(node_id)
            if not node:
                continue

            # Skip human nodes - they wait for external input
            if node.type == NodeType.HUMAN:
                continue

            try:
                output = await self.reducer.execute_node(definition, state, node_id, context)

                # Create completion event
                event = WorkflowEvent(
                    event_type="node_complete",
                    node_id=node_id,
                    data=output,
                )
                state = await self.reducer.reduce(definition, state, event, context)

            except Exception as e:
                # Create failure event
                event = WorkflowEvent(
                    event_type="node_failed",
                    node_id=node_id,
                    data={"error": str(e)},
                )
                state = await self.reducer.reduce(definition, state, event, context)
                break

        return state

    async def run_to_completion(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        context: dict[str, Any] | None = None,
        max_steps: int = 100,
    ) -> WorkflowState:
        """
        Run workflow until completion or pause.

        Factor 8: Own your control flow.
        """
        context = context or {}
        steps = 0

        while state.status == WorkflowStatus.RUNNING and steps < max_steps:
            state = await self.step(definition, state, context)
            steps += 1

        return state

    def pause(self, state: WorkflowState) -> WorkflowState:
        """Pause workflow execution. Factor 6."""
        event = WorkflowEvent(event_type="pause")
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.reducer.reduce(WorkflowDefinition("", "", [], []), state, event)
        )

    def resume(self, state: WorkflowState) -> WorkflowState:
        """Resume workflow execution. Factor 6."""
        event = WorkflowEvent(event_type="resume")
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.reducer.reduce(WorkflowDefinition("", "", [], []), state, event)
        )

    async def provide_human_response(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState,
        node_id: str,
        response: Any,
        context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """
        Provide human response to a waiting node.

        Factor 7: Human-in-the-loop.
        """
        event = WorkflowEvent(
            event_type="human_response",
            node_id=node_id,
            data={"response": response},
        )
        state = await self.reducer.reduce(definition, state, event, context)

        # Continue execution
        return await self._execute_current_nodes(definition, state, context or {})


# Builder for creating workflows declaratively
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
