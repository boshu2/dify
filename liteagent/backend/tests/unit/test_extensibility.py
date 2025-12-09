"""
Tests for extensibility and plugin architecture.

Validates:
- Provider registry pattern
- Plugin loading
- Workflow extension points
- Custom handlers and hooks
"""
import pytest
from unittest.mock import MagicMock, AsyncMock

from app.core.registry import (
    ProviderRegistry,
    ProviderMetadata,
    LLMProvider,
    EmbeddingProvider,
    VectorStoreProvider,
    StorageProvider,
)


# =============================================================================
# Provider Registry Tests
# =============================================================================


class TestProviderRegistry:
    """Tests for generic provider registry."""

    def test_register_decorator(self):
        """Should register provider via decorator."""
        registry: ProviderRegistry[LLMProvider] = ProviderRegistry("test")

        @registry.register("test_provider", description="Test provider")
        class TestProvider(LLMProvider):
            async def chat(self, messages, tools=None, **kwargs):
                return {}

            async def complete(self, prompt, **kwargs):
                return ""

        assert "test_provider" in registry.list()
        assert registry.has("test_provider")

    def test_register_factory(self):
        """Should register provider factory."""
        registry: ProviderRegistry = ProviderRegistry("test")

        def create_provider(**config):
            return MagicMock(**config)

        registry.register_factory("factory_provider", create_provider, description="Factory")

        assert "factory_provider" in registry.list()
        provider = registry.get("factory_provider", key="value")
        assert provider.key == "value"

    def test_register_instance(self):
        """Should register pre-instantiated provider."""
        registry: ProviderRegistry = ProviderRegistry("test")
        instance = MagicMock()

        registry.register_instance("instance_provider", instance)

        assert registry.get("instance_provider") is instance

    def test_get_default(self):
        """Should return default provider."""
        registry: ProviderRegistry = ProviderRegistry("test")

        @registry.register("first", is_default=True)
        class First:
            pass

        @registry.register("second")
        class Second:
            pass

        assert registry.get_default() == "first"
        assert isinstance(registry.get(), First)

    def test_set_default(self):
        """Should change default provider."""
        registry: ProviderRegistry = ProviderRegistry("test")

        @registry.register("a")
        class A:
            pass

        @registry.register("b")
        class B:
            pass

        registry.set_default("b")
        assert registry.get_default() == "b"

    def test_get_unknown_raises(self):
        """Should raise KeyError for unknown provider."""
        registry: ProviderRegistry = ProviderRegistry("test")

        with pytest.raises(KeyError) as exc:
            registry.get("nonexistent")

        assert "nonexistent" in str(exc.value)

    def test_metadata(self):
        """Should store and retrieve metadata."""
        registry: ProviderRegistry = ProviderRegistry("test")

        @registry.register(
            "meta_provider",
            description="With metadata",
            version="2.0.0",
            tags=["cloud", "fast"],
        )
        class MetaProvider:
            pass

        meta = registry.get_metadata("meta_provider")
        assert meta is not None
        assert meta.description == "With metadata"
        assert meta.version == "2.0.0"
        assert "cloud" in meta.tags

    def test_list_with_metadata(self):
        """Should list all providers with metadata."""
        registry: ProviderRegistry = ProviderRegistry("test")

        @registry.register("p1", description="Provider 1")
        class P1:
            pass

        @registry.register("p2", description="Provider 2")
        class P2:
            pass

        all_meta = registry.list_with_metadata()
        assert len(all_meta) == 2
        names = [m.name for m in all_meta]
        assert "p1" in names
        assert "p2" in names

    def test_unregister(self):
        """Should unregister provider."""
        registry: ProviderRegistry = ProviderRegistry("test")

        @registry.register("temp")
        class Temp:
            pass

        assert registry.has("temp")
        registry.unregister("temp")
        assert not registry.has("temp")

    def test_clear(self):
        """Should clear all providers."""
        registry: ProviderRegistry = ProviderRegistry("test")

        @registry.register("p1")
        class P1:
            pass

        @registry.register("p2")
        class P2:
            pass

        registry.clear()
        assert len(registry.list()) == 0


class TestProviderInterfaces:
    """Tests for provider interface contracts."""

    def test_llm_provider_interface(self):
        """LLMProvider should define required methods."""
        assert hasattr(LLMProvider, "chat")
        assert hasattr(LLMProvider, "complete")

    def test_embedding_provider_interface(self):
        """EmbeddingProvider should define required methods."""
        assert hasattr(EmbeddingProvider, "embed")
        assert hasattr(EmbeddingProvider, "get_dimension")

    def test_vector_store_provider_interface(self):
        """VectorStoreProvider should define required methods."""
        assert hasattr(VectorStoreProvider, "add")
        assert hasattr(VectorStoreProvider, "search")
        assert hasattr(VectorStoreProvider, "delete")

    def test_storage_provider_interface(self):
        """StorageProvider should define required methods."""
        assert hasattr(StorageProvider, "get")
        assert hasattr(StorageProvider, "set")
        assert hasattr(StorageProvider, "delete")
        assert hasattr(StorageProvider, "exists")


# =============================================================================
# Workflow Extension Tests
# =============================================================================


class TestWorkflowHooks:
    """Tests for workflow execution hooks."""

    @pytest.fixture
    def hooks(self):
        from app.workflows.extensions import WorkflowHooks
        return WorkflowHooks()

    @pytest.fixture
    def mock_node(self):
        from app.workflows.types import NodeDefinition, NodeType
        return NodeDefinition(id="test", type=NodeType.AGENT)

    @pytest.fixture
    def mock_state(self):
        from app.workflows.state import WorkflowState
        return WorkflowState(workflow_id="test", execution_id="exec1")

    @pytest.mark.asyncio
    async def test_before_node_hook(self, hooks, mock_node, mock_state):
        """Should execute before-node hooks."""
        called = []

        @hooks.before_node("agent")
        async def before_agent(node, state):
            called.append("agent")
            return state

        await hooks.run_before_node(mock_node, mock_state)
        assert "agent" in called

    @pytest.mark.asyncio
    async def test_after_node_hook(self, hooks, mock_node, mock_state):
        """Should execute after-node hooks."""
        called = []

        @hooks.after_node("agent")
        async def after_agent(node, state, output):
            called.append("after")
            output["modified"] = True
            return output

        result = await hooks.run_after_node(mock_node, mock_state, {"original": True})
        assert "after" in called
        assert result.get("modified") is True

    @pytest.mark.asyncio
    async def test_global_hook(self, hooks, mock_node, mock_state):
        """Should execute global hooks for all node types."""
        called = []

        @hooks.before_node(None)  # Global hook
        async def global_hook(node, state):
            called.append("global")
            return state

        await hooks.run_before_node(mock_node, mock_state)
        assert "global" in called

    @pytest.mark.asyncio
    async def test_error_hook_suppress(self, hooks, mock_node, mock_state):
        """Should allow hooks to suppress errors."""

        @hooks.on_error("agent")
        async def handle_error(node, state, error):
            return True  # Suppress

        should_suppress = await hooks.run_on_error(
            mock_node, mock_state, ValueError("test error")
        )
        assert should_suppress is True

    @pytest.mark.asyncio
    async def test_workflow_lifecycle_hooks(self, hooks, mock_state):
        """Should execute workflow lifecycle hooks."""
        called = []

        @hooks.before_workflow_start
        async def before_start(state):
            called.append("before")

        @hooks.after_workflow_complete
        async def after_complete(state):
            called.append("after")

        await hooks.run_before_workflow(mock_state)
        await hooks.run_after_workflow(mock_state)

        assert called == ["before", "after"]


class TestConditionRegistry:
    """Tests for custom condition evaluators."""

    @pytest.fixture
    def registry(self):
        from app.workflows.extensions import ConditionRegistry
        return ConditionRegistry()

    def test_register_evaluator(self, registry):
        """Should register custom evaluator."""

        @registry.register("custom:")
        def custom_eval(expr, context):
            return expr == "true"

        assert registry.evaluate("custom:true", {}) is True
        assert registry.evaluate("custom:false", {}) is False

    def test_default_python_eval(self, registry):
        """Should fall back to Python eval."""
        context = {"x": 10, "y": 5}
        assert registry.evaluate("x > y", context) is True
        assert registry.evaluate("x < y", context) is False

    def test_builtin_contains(self):
        """Should have built-in contains evaluator."""
        from app.workflows.extensions import condition_registry

        context = {"message": "hello world"}
        assert condition_registry.evaluate("contains:message, world", context) is True
        assert condition_registry.evaluate("contains:message, foo", context) is False

    def test_builtin_equals(self):
        """Should have built-in equals evaluator."""
        from app.workflows.extensions import condition_registry

        context = {"status": "active"}
        assert condition_registry.evaluate("equals:status, active", context) is True
        assert condition_registry.evaluate("equals:status, inactive", context) is False

    def test_builtin_regex(self):
        """Should have built-in regex evaluator."""
        from app.workflows.extensions import condition_registry

        context = {"email": "test@example.com"}
        assert condition_registry.evaluate(r"regex:email, .*@.*\.com", context) is True
        assert condition_registry.evaluate(r"regex:email, .*@.*\.org", context) is False


class TestNodeHandlerRegistry:
    """Tests for custom node handler registration."""

    def test_register_custom_handler(self):
        """Should register custom node handler."""
        from app.workflows.extensions import node_handler_registry
        from app.workflows.handlers import NodeHandler

        @node_handler_registry.register("custom_node", description="Custom node")
        class CustomHandler(NodeHandler):
            async def execute(self, node, state, context):
                return {"custom": True}

        assert node_handler_registry.has("custom_node")
        handler = node_handler_registry.get("custom_node")
        assert handler is not None


class TestTransformerRegistry:
    """Tests for state transformers."""

    @pytest.fixture
    def registry(self):
        from app.workflows.extensions import TransformerRegistry
        return TransformerRegistry()

    @pytest.fixture
    def mock_state(self):
        from app.workflows.state import WorkflowState
        return WorkflowState(workflow_id="test", execution_id="exec1")

    def test_register_transformer(self, registry, mock_state):
        """Should register and apply transformer."""

        @registry.register("add_metadata")
        def add_metadata(state):
            state.metadata["transformed"] = True
            return state

        result = registry.apply("add_metadata", mock_state)
        assert result.metadata.get("transformed") is True

    def test_list_transformers(self, registry):
        """Should list all registered transformers."""

        @registry.register("t1")
        def t1(state):
            return state

        @registry.register("t2")
        def t2(state):
            return state

        names = registry.list()
        assert "t1" in names
        assert "t2" in names


# =============================================================================
# Plugin Loading Tests
# =============================================================================


class TestPluginLoading:
    """Tests for plugin loading system."""

    def test_load_all_plugins(self):
        """Should load all built-in plugins."""
        from app.core.plugins import load_all_plugins, get_provider_summary

        load_all_plugins()

        summary = get_provider_summary()

        # Check LLM providers are registered
        assert "llm" in summary
        assert "openai" in summary["llm"]

        # Check embedding providers
        assert "embedding" in summary
        assert "nemotron" in summary["embedding"]

        # Check storage providers
        assert "storage" in summary
        assert "memory" in summary["storage"]

    def test_get_provider_after_load(self):
        """Should get provider after loading plugins."""
        from app.core.plugins import load_all_plugins
        from app.core.registry import storage_registry

        load_all_plugins()

        # Get memory storage (doesn't need external deps)
        storage = storage_registry.get("memory")
        assert storage is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestExtensibilityIntegration:
    """Integration tests for extensibility features."""

    @pytest.mark.asyncio
    async def test_custom_provider_workflow(self):
        """Should use custom provider in workflow."""
        from app.core.registry import ProviderRegistry, LLMProvider

        # Create custom registry
        test_registry: ProviderRegistry[LLMProvider] = ProviderRegistry("test_llm")

        # Register mock provider
        class MockLLM(LLMProvider):
            async def chat(self, messages, tools=None, **kwargs):
                return {
                    "choices": [{"message": {"content": "Mock response", "role": "assistant"}}]
                }

            async def complete(self, prompt, **kwargs):
                return "Mock completion"

        test_registry.register_instance("mock", MockLLM())

        # Use provider
        provider = test_registry.get("mock")
        response = await provider.chat([{"role": "user", "content": "Hello"}])

        assert response["choices"][0]["message"]["content"] == "Mock response"

    @pytest.mark.asyncio
    async def test_hooks_modify_execution(self):
        """Should allow hooks to modify workflow execution."""
        from app.workflows.extensions import WorkflowHooks
        from app.workflows.types import NodeDefinition, NodeType
        from app.workflows.state import WorkflowState

        hooks = WorkflowHooks()
        modifications = []

        @hooks.before_node("transform")
        async def inject_data(node, state):
            state.set_variable("injected", "from_hook")
            modifications.append("before")
            return state

        @hooks.after_node("transform")
        async def log_output(node, state, output):
            modifications.append("after")
            output["logged"] = True
            return output

        node = NodeDefinition(id="t1", type=NodeType.TRANSFORM)
        state = WorkflowState(workflow_id="w1", execution_id="e1")

        state = await hooks.run_before_node(node, state)
        output = await hooks.run_after_node(node, state, {"result": "data"})

        assert state.get_variable("injected") == "from_hook"
        assert output.get("logged") is True
        assert modifications == ["before", "after"]
