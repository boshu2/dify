"""
Workflow Engine - orchestration layer.

Implements Factor 6 (Launch/Pause/Resume), Factor 8 (Own your control flow),
and Factor 11 (Entry point for triggering workflows).

This module only contains the WorkflowEngine orchestrator.
Types, state, handlers, and reducer are in separate modules.
"""
import uuid
from typing import Any

from app.workflows.types import NodeType, WorkflowEvent, WorkflowStatus
from app.workflows.state import WorkflowDefinition, WorkflowState
from app.workflows.reducer import WorkflowReducer


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


# Re-export for backward compatibility
from app.workflows.types import (  # noqa: E402, F401
    NodeType,
    WorkflowStatus,
    NodeStatus,
    Edge,
    NodeDefinition,
    NodeExecution,
    WorkflowEvent,
)
from app.workflows.state import WorkflowDefinition, WorkflowState  # noqa: E402, F401
from app.workflows.handlers import NodeHandler  # noqa: E402, F401
from app.workflows.reducer import WorkflowReducer  # noqa: E402, F401
from app.workflows.builder import WorkflowBuilder  # noqa: E402, F401
