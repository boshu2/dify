"""
Workflow Extension Points.

Provides pluggable extension points for:
- Custom node handlers
- Pre/post execution hooks
- Custom condition evaluators
- State transformers

Usage:
    # Register a custom node handler
    @node_handler_registry.register("custom_api")
    class CustomAPIHandler(NodeHandler):
        async def execute(self, node, state, context):
            ...

    # Register execution hooks
    @workflow_hooks.before_node("agent")
    async def log_before_agent(node, state):
        logger.info(f"Executing agent: {node.id}")

    @workflow_hooks.after_node("agent")
    async def log_after_agent(node, state, result):
        logger.info(f"Agent completed: {result}")
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from app.core.registry import ProviderRegistry
from app.workflows.types import NodeType, NodeDefinition
from app.workflows.state import WorkflowState
from app.workflows.handlers import NodeHandler

logger = logging.getLogger(__name__)


# =============================================================================
# Node Handler Registry
# =============================================================================

node_handler_registry: ProviderRegistry[NodeHandler] = ProviderRegistry("node_handler")


def register_node_handler(
    node_type: str,
    description: str = "",
) -> Callable[[type[NodeHandler]], type[NodeHandler]]:
    """
    Decorator to register a custom node handler.

    Usage:
        @register_node_handler("http_request")
        class HTTPRequestHandler(NodeHandler):
            async def execute(self, node, state, context):
                ...
    """
    return node_handler_registry.register(node_type, description=description)


# =============================================================================
# Execution Hooks
# =============================================================================

BeforeNodeHook = Callable[[NodeDefinition, WorkflowState], Awaitable[WorkflowState | None]]
AfterNodeHook = Callable[[NodeDefinition, WorkflowState, dict[str, Any]], Awaitable[dict[str, Any] | None]]
ErrorHook = Callable[[NodeDefinition, WorkflowState, Exception], Awaitable[bool]]


@dataclass
class WorkflowHooks:
    """
    Registry for workflow execution hooks.

    Hooks allow extending workflow behavior without modifying core logic.
    """

    _before_node: dict[str, list[BeforeNodeHook]] = field(default_factory=dict)
    _after_node: dict[str, list[AfterNodeHook]] = field(default_factory=dict)
    _on_error: dict[str, list[ErrorHook]] = field(default_factory=dict)
    _before_workflow: list[Callable[[WorkflowState], Awaitable[None]]] = field(default_factory=list)
    _after_workflow: list[Callable[[WorkflowState], Awaitable[None]]] = field(default_factory=list)

    def before_node(
        self,
        node_type: str | None = None,
    ) -> Callable[[BeforeNodeHook], BeforeNodeHook]:
        """
        Register hook to run before node execution.

        Args:
            node_type: Specific node type to hook, or None for all nodes

        The hook can modify state before execution or return new state.
        """
        key = node_type or "*"

        def decorator(func: BeforeNodeHook) -> BeforeNodeHook:
            if key not in self._before_node:
                self._before_node[key] = []
            self._before_node[key].append(func)
            return func

        return decorator

    def after_node(
        self,
        node_type: str | None = None,
    ) -> Callable[[AfterNodeHook], AfterNodeHook]:
        """
        Register hook to run after node execution.

        The hook can modify the node output before it's processed.
        """
        key = node_type or "*"

        def decorator(func: AfterNodeHook) -> AfterNodeHook:
            if key not in self._after_node:
                self._after_node[key] = []
            self._after_node[key].append(func)
            return func

        return decorator

    def on_error(
        self,
        node_type: str | None = None,
    ) -> Callable[[ErrorHook], ErrorHook]:
        """
        Register hook for error handling.

        The hook can return True to suppress the error and continue.
        """
        key = node_type or "*"

        def decorator(func: ErrorHook) -> ErrorHook:
            if key not in self._on_error:
                self._on_error[key] = []
            self._on_error[key].append(func)
            return func

        return decorator

    def before_workflow_start(
        self,
        func: Callable[[WorkflowState], Awaitable[None]],
    ) -> Callable[[WorkflowState], Awaitable[None]]:
        """Register hook to run before workflow starts."""
        self._before_workflow.append(func)
        return func

    def after_workflow_complete(
        self,
        func: Callable[[WorkflowState], Awaitable[None]],
    ) -> Callable[[WorkflowState], Awaitable[None]]:
        """Register hook to run after workflow completes."""
        self._after_workflow.append(func)
        return func

    async def run_before_node(
        self,
        node: NodeDefinition,
        state: WorkflowState,
    ) -> WorkflowState:
        """Execute all before-node hooks."""
        # Run type-specific hooks
        node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
        for hook in self._before_node.get(node_type, []):
            result = await hook(node, state)
            if result is not None:
                state = result

        # Run global hooks
        for hook in self._before_node.get("*", []):
            result = await hook(node, state)
            if result is not None:
                state = result

        return state

    async def run_after_node(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        output: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute all after-node hooks."""
        node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
        for hook in self._after_node.get(node_type, []):
            result = await hook(node, state, output)
            if result is not None:
                output = result

        for hook in self._after_node.get("*", []):
            result = await hook(node, state, output)
            if result is not None:
                output = result

        return output

    async def run_on_error(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        error: Exception,
    ) -> bool:
        """
        Execute error hooks.

        Returns True if error should be suppressed.
        """
        node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
        for hook in self._on_error.get(node_type, []):
            if await hook(node, state, error):
                return True

        for hook in self._on_error.get("*", []):
            if await hook(node, state, error):
                return True

        return False

    async def run_before_workflow(self, state: WorkflowState) -> None:
        """Execute all before-workflow hooks."""
        for hook in self._before_workflow:
            await hook(state)

    async def run_after_workflow(self, state: WorkflowState) -> None:
        """Execute all after-workflow hooks."""
        for hook in self._after_workflow:
            await hook(state)


# Global hooks instance
workflow_hooks = WorkflowHooks()


# =============================================================================
# Condition Evaluators
# =============================================================================

ConditionEvaluator = Callable[[str, dict[str, Any]], bool]


class ConditionRegistry:
    """
    Registry for custom condition evaluators.

    Allows extending the condition evaluation beyond simple Python expressions.
    """

    def __init__(self):
        self._evaluators: dict[str, ConditionEvaluator] = {}

    def register(
        self,
        prefix: str,
    ) -> Callable[[ConditionEvaluator], ConditionEvaluator]:
        """
        Register a condition evaluator for expressions starting with prefix.

        Usage:
            @condition_registry.register("jmespath:")
            def jmespath_evaluator(expr: str, context: dict) -> bool:
                import jmespath
                return bool(jmespath.search(expr, context))
        """
        def decorator(func: ConditionEvaluator) -> ConditionEvaluator:
            self._evaluators[prefix] = func
            return func
        return decorator

    def evaluate(self, condition: str, context: dict[str, Any]) -> bool:
        """
        Evaluate a condition using registered evaluators.

        Falls back to safe_eval for simple expressions.
        """
        from app.core.safe_eval import safe_eval_condition

        for prefix, evaluator in self._evaluators.items():
            if condition.startswith(prefix):
                return evaluator(condition[len(prefix):], context)

        # Default: Safe expression evaluation
        try:
            return safe_eval_condition(condition, context)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {condition}, error: {e}")
            return False


condition_registry = ConditionRegistry()


# =============================================================================
# State Transformers
# =============================================================================

StateTransformer = Callable[[WorkflowState], WorkflowState]


class TransformerRegistry:
    """
    Registry for state transformers.

    Transformers can modify state at various points in the workflow.
    """

    def __init__(self):
        self._transformers: dict[str, StateTransformer] = {}

    def register(self, name: str) -> Callable[[StateTransformer], StateTransformer]:
        """Register a named state transformer."""
        def decorator(func: StateTransformer) -> StateTransformer:
            self._transformers[name] = func
            return func
        return decorator

    def get(self, name: str) -> StateTransformer | None:
        """Get a transformer by name."""
        return self._transformers.get(name)

    def apply(self, name: str, state: WorkflowState) -> WorkflowState:
        """Apply a named transformer to state."""
        transformer = self._transformers.get(name)
        if transformer:
            return transformer(state)
        return state

    def list(self) -> list[str]:
        """List all registered transformer names."""
        return list(self._transformers.keys())


transformer_registry = TransformerRegistry()


# =============================================================================
# Built-in Extensions
# =============================================================================

# Register built-in condition evaluators
@condition_registry.register("contains:")
def contains_evaluator(expr: str, context: dict[str, Any]) -> bool:
    """Check if a value contains a substring."""
    parts = expr.split(",", 1)
    if len(parts) != 2:
        return False
    var_name, substring = parts[0].strip(), parts[1].strip()
    value = context.get(var_name, "")
    return substring in str(value)


@condition_registry.register("equals:")
def equals_evaluator(expr: str, context: dict[str, Any]) -> bool:
    """Check if a value equals another."""
    parts = expr.split(",", 1)
    if len(parts) != 2:
        return False
    var_name, expected = parts[0].strip(), parts[1].strip()
    value = str(context.get(var_name, ""))
    return value == expected


@condition_registry.register("regex:")
def regex_evaluator(expr: str, context: dict[str, Any]) -> bool:
    """Check if a value matches a regex pattern."""
    import re
    parts = expr.split(",", 1)
    if len(parts) != 2:
        return False
    var_name, pattern = parts[0].strip(), parts[1].strip()
    value = str(context.get(var_name, ""))
    return bool(re.search(pattern, value))
