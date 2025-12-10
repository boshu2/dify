"""
Workflow reducer - stateless state machine.

Implements Factor 12: Pure function that transforms state based on events.
Implements Factor 8: Owns control flow through explicit state transitions.
"""
from datetime import datetime, timezone
from typing import Any

from app.workflows.types import (
    NodeStatus,
    NodeType,
    WorkflowEvent,
    WorkflowStatus,
)
from app.workflows.state import WorkflowDefinition, WorkflowState
from app.workflows.handlers import (
    NodeHandler,
    StartNodeHandler,
    EndNodeHandler,
    AgentNodeHandler,
    ConditionNodeHandler,
    TransformNodeHandler,
    HumanNodeHandler,
    HTTPRequestNodeHandler,
    LLMNodeHandler,
    CodeNodeHandler,
    KnowledgeRetrievalNodeHandler,
    LoopNodeHandler,
    VariableAggregatorNodeHandler,
    ParallelNodeHandler,
    MergeNodeHandler,
    WaitNodeHandler,
)


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
            NodeType.HTTP_REQUEST: HTTPRequestNodeHandler(),
            NodeType.LLM: LLMNodeHandler(),
            NodeType.CODE: CodeNodeHandler(),
            NodeType.KNOWLEDGE_RETRIEVAL: KnowledgeRetrievalNodeHandler(),
            NodeType.LOOP: LoopNodeHandler(),
            NodeType.VARIABLE_AGGREGATOR: VariableAggregatorNodeHandler(),
            NodeType.PARALLEL: ParallelNodeHandler(),
            NodeType.MERGE: MergeNodeHandler(),
            NodeType.WAIT: WaitNodeHandler(),
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
        from app.core.safe_eval import safe_eval_condition

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
                    if safe_eval_condition(edge.condition, eval_context):
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
