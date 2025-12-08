"""
Workflow node handlers.

Each handler is responsible for executing a specific type of workflow node.
"""
import json
from abc import ABC, abstractmethod
from typing import Any

from app.workflows.types import NodeDefinition
from app.workflows.state import WorkflowState


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
