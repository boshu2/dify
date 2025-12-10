"""
Tests for sandbox code execution features.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.sandbox.config import SandboxConfig, get_sandbox_config
from app.core.sandbox.executor import (
    CodeExecutor,
    CodeLanguage,
    ExecutionError,
    ExecutionResult,
    get_executor,
)
from app.core.sandbox.plugins import (
    Plugin,
    PluginInstallError,
    PluginManager,
    PluginMetadata,
    PluginStatus,
    PluginType,
    get_plugin_manager,
)
from app.core.sandbox.transformers import (
    CodeTransformer,
    JavaScriptTransformer,
    Jinja2Transformer,
    PythonTransformer,
)
from app.core.sandbox.validator import OutputValidator, ValidationError, ValidationResult


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()

        assert config.mode == "local"
        assert config.execution_timeout == 30
        assert config.max_workers == 4
        assert config.max_memory_mb == 256
        assert config.max_output_size == 1_000_000
        assert config.enable_network is False

    def test_config_allowed_modules(self):
        """Test default allowed modules."""
        config = SandboxConfig()

        assert "json" in config.allowed_modules
        assert "math" in config.allowed_modules
        assert "random" in config.allowed_modules
        assert "datetime" in config.allowed_modules

    def test_config_blocked_patterns(self):
        """Test default blocked patterns."""
        config = SandboxConfig()

        assert "import os" in config.blocked_patterns
        assert "import subprocess" in config.blocked_patterns
        assert "__import__" in config.blocked_patterns
        assert "eval(" in config.blocked_patterns

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = SandboxConfig(
            mode="docker",
            execution_timeout=60,
            max_workers=8,
            enable_network=True,
        )

        assert config.mode == "docker"
        assert config.execution_timeout == 60
        assert config.max_workers == 8
        assert config.enable_network is True

    def test_get_sandbox_config_singleton(self):
        """Test singleton pattern for config."""
        config1 = get_sandbox_config()
        config2 = get_sandbox_config()

        assert config1 is config2


class TestCodeLanguage:
    """Tests for CodeLanguage enum."""

    def test_language_values(self):
        """Test language enum values."""
        assert CodeLanguage.PYTHON3.value == "python3"
        assert CodeLanguage.JAVASCRIPT.value == "javascript"
        assert CodeLanguage.JINJA2.value == "jinja2"

    def test_language_from_string(self):
        """Test creating language from string."""
        lang = CodeLanguage("python3")
        assert lang == CodeLanguage.PYTHON3

        lang = CodeLanguage("javascript")
        assert lang == CodeLanguage.JAVASCRIPT

    def test_invalid_language(self):
        """Test invalid language raises error."""
        with pytest.raises(ValueError):
            CodeLanguage("invalid")


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_success_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            output={"key": "value"},
            stdout="output line",
            stderr="",
            execution_time_ms=100.5,
        )

        assert result.success is True
        assert result.output == {"key": "value"}
        assert result.stdout == "output line"
        assert result.error is None

    def test_error_result(self):
        """Test error execution result."""
        result = ExecutionResult(
            success=False,
            error="Syntax error",
            error_type="SyntaxError",
            stderr="line 1: syntax error",
            execution_time_ms=50.0,
        )

        assert result.success is False
        assert result.error == "Syntax error"
        assert result.error_type == "SyntaxError"


class TestPythonTransformer:
    """Tests for PythonTransformer."""

    def test_transform_simple_code(self):
        """Test transforming simple Python code."""
        transformer = PythonTransformer()
        code = "result = x + y"
        inputs = {"x": 1, "y": 2}

        transformed, error = transformer.transform(code, inputs)

        assert error is None
        assert "import json" in transformed
        assert "import base64" in transformed
        assert "x + y" in transformed

    def test_transform_with_preload(self):
        """Test transforming code with preload."""
        transformer = PythonTransformer()
        code = "result = helper(x)"
        inputs = {"x": 5}
        preload = "def helper(n): return n * 2"

        transformed, error = transformer.transform(code, inputs, preload)

        assert error is None
        assert "helper(n)" in transformed
        assert "helper(x)" in transformed

    def test_transform_detects_blocked_pattern(self):
        """Test transformer detects blocked patterns."""
        transformer = PythonTransformer()
        code = "import os\nos.system('rm -rf /')"
        inputs = {}

        # The transformer itself doesn't block, the executor does
        transformed, error = transformer.transform(code, inputs)
        assert transformed is not None  # Transform succeeds, blocking is at execution

    def test_transform_empty_inputs(self):
        """Test transforming with empty inputs."""
        transformer = PythonTransformer()
        code = "result = 42"
        inputs = {}

        transformed, error = transformer.transform(code, inputs)

        assert error is None
        assert "result = 42" in transformed


class TestJavaScriptTransformer:
    """Tests for JavaScriptTransformer."""

    def test_transform_simple_code(self):
        """Test transforming simple JavaScript code."""
        transformer = JavaScriptTransformer()
        code = "const result = x + y;"
        inputs = {"x": 1, "y": 2}

        transformed, error = transformer.transform(code, inputs)

        assert error is None
        assert "JSON.parse" in transformed
        assert "const result = x + y" in transformed

    def test_transform_with_preload(self):
        """Test transforming JavaScript with preload."""
        transformer = JavaScriptTransformer()
        code = "const result = helper(x);"
        inputs = {"x": 5}
        preload = "function helper(n) { return n * 2; }"

        transformed, error = transformer.transform(code, inputs, preload)

        assert error is None
        assert "function helper" in transformed

    def test_transform_json_output(self):
        """Test JavaScript output is JSON encoded."""
        transformer = JavaScriptTransformer()
        code = "const result = {key: 'value'};"
        inputs = {}

        transformed, error = transformer.transform(code, inputs)

        assert error is None
        assert "JSON.stringify" in transformed


class TestJinja2Transformer:
    """Tests for Jinja2Transformer."""

    def test_transform_simple_template(self):
        """Test transforming simple Jinja2 template."""
        transformer = Jinja2Transformer()
        template = "Hello, {{ name }}!"
        inputs = {"name": "World"}

        transformed, error = transformer.transform(template, inputs)

        assert error is None
        assert "Hello, {{ name }}!" in transformed

    def test_transform_with_loop(self):
        """Test transforming template with loop."""
        transformer = Jinja2Transformer()
        template = "{% for item in items %}{{ item }}{% endfor %}"
        inputs = {"items": [1, 2, 3]}

        transformed, error = transformer.transform(template, inputs)

        assert error is None
        assert "{% for item in items %}" in transformed


class TestOutputValidator:
    """Tests for OutputValidator."""

    def test_validate_string(self):
        """Test validating string output."""
        validator = OutputValidator()
        result = validator.validate("hello world")

        assert result.valid is True
        assert result.value == "hello world"

    def test_validate_number(self):
        """Test validating number output."""
        validator = OutputValidator()

        result = validator.validate(42)
        assert result.valid is True
        assert result.value == 42

        result = validator.validate(3.14)
        assert result.valid is True
        assert result.value == 3.14

    def test_validate_dict(self):
        """Test validating dict output."""
        validator = OutputValidator()
        data = {"key": "value", "nested": {"a": 1}}
        result = validator.validate(data)

        assert result.valid is True
        assert result.value == data

    def test_validate_list(self):
        """Test validating list output."""
        validator = OutputValidator()
        data = [1, 2, 3, "four"]
        result = validator.validate(data)

        assert result.valid is True
        assert result.value == data

    def test_validate_none(self):
        """Test validating None output."""
        validator = OutputValidator()
        result = validator.validate(None)

        assert result.valid is True
        assert result.value is None

    def test_validate_bool(self):
        """Test validating bool output."""
        validator = OutputValidator()

        result = validator.validate(True)
        assert result.valid is True
        assert result.value is True

        result = validator.validate(False)
        assert result.valid is True
        assert result.value is False

    def test_validate_string_too_long(self):
        """Test string exceeding max length - gets truncated with warning."""
        config = SandboxConfig(max_string_length=10)
        validator = OutputValidator(config)
        result = validator.validate("x" * 100)

        # String gets truncated, not rejected
        assert result.valid is True
        assert len(result.value) == 10
        assert result.warnings is not None
        assert any("truncated" in w.lower() for w in result.warnings)

    def test_validate_array_too_long(self):
        """Test array exceeding max length - gets truncated with warning."""
        config = SandboxConfig(max_array_length=5)
        validator = OutputValidator(config)
        result = validator.validate(list(range(100)))

        # Array gets truncated, not rejected
        assert result.valid is True
        assert len(result.value) == 5
        assert result.warnings is not None
        assert any("truncated" in w.lower() for w in result.warnings)

    def test_validate_number_out_of_range(self):
        """Test number out of range."""
        config = SandboxConfig(max_number=1000)
        validator = OutputValidator(config)
        result = validator.validate(10000)

        assert result.valid is False
        assert "exceeds maximum" in result.error

    def test_validate_dict_too_deep(self):
        """Test dict exceeding max depth."""
        config = SandboxConfig(max_depth=2)
        validator = OutputValidator(config)
        data = {"a": {"b": {"c": {"d": 1}}}}
        result = validator.validate(data)

        assert result.valid is False
        assert "depth" in result.error.lower()


class TestPluginMetadata:
    """Tests for PluginMetadata."""

    def test_create_metadata(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
        )

        assert metadata.id == "test-plugin"
        assert metadata.name == "Test Plugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "A test plugin"
        assert metadata.plugin_type == PluginType.TOOL

    def test_metadata_to_dict(self):
        """Test converting metadata to dict."""
        metadata = PluginMetadata(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
        )

        data = metadata.to_dict()

        assert data["id"] == "test-plugin"
        assert data["name"] == "Test Plugin"
        assert data["version"] == "1.0.0"
        assert "created_at" in data

    def test_metadata_from_dict(self):
        """Test creating metadata from dict."""
        data = {
            "id": "test-plugin",
            "name": "Test Plugin",
            "version": "2.0.0",
            "plugin_type": "provider",
            "tags": ["ai", "llm"],
        }

        metadata = PluginMetadata.from_dict(data)

        assert metadata.id == "test-plugin"
        assert metadata.name == "Test Plugin"
        assert metadata.version == "2.0.0"
        assert metadata.plugin_type == PluginType.PROVIDER
        assert metadata.tags == ["ai", "llm"]


class TestPlugin:
    """Tests for Plugin."""

    def test_create_plugin(self):
        """Test creating a plugin."""
        metadata = PluginMetadata(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
        )
        plugin = Plugin(metadata=metadata)

        assert plugin.metadata.id == "test-plugin"
        assert plugin.status == PluginStatus.INSTALLED
        assert plugin.config == {}

    def test_plugin_to_dict(self):
        """Test converting plugin to dict."""
        metadata = PluginMetadata(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
        )
        plugin = Plugin(
            metadata=metadata,
            status=PluginStatus.ENABLED,
            install_path="/path/to/plugin",
        )

        data = plugin.to_dict()

        assert data["metadata"]["id"] == "test-plugin"
        assert data["status"] == "enabled"
        assert data["install_path"] == "/path/to/plugin"


class TestPluginStatus:
    """Tests for PluginStatus enum."""

    def test_status_values(self):
        """Test plugin status values."""
        assert PluginStatus.AVAILABLE.value == "available"
        assert PluginStatus.INSTALLED.value == "installed"
        assert PluginStatus.ENABLED.value == "enabled"
        assert PluginStatus.DISABLED.value == "disabled"
        assert PluginStatus.ERROR.value == "error"


class TestPluginType:
    """Tests for PluginType enum."""

    def test_type_values(self):
        """Test plugin type values."""
        assert PluginType.TOOL.value == "tool"
        assert PluginType.PROVIDER.value == "provider"
        assert PluginType.EXTRACTOR.value == "extractor"
        assert PluginType.TRANSFORMER.value == "transformer"
        assert PluginType.MIDDLEWARE.value == "middleware"


class TestPluginManager:
    """Tests for PluginManager."""

    def test_create_manager(self):
        """Test creating a plugin manager."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            assert manager.plugins_dir.exists()
            assert manager._plugins == {}

    def test_list_plugins_empty(self):
        """Test listing plugins when none installed."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            plugins = manager.list_plugins()
            assert plugins == []

    def test_get_plugin_not_found(self):
        """Test getting non-existent plugin."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            plugin = manager.get_plugin("non-existent")
            assert plugin is None

    def test_get_tools_empty(self):
        """Test getting tools when none registered."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            tools = manager.get_tools()
            assert tools == {}

    @pytest.mark.asyncio
    async def test_install_from_directory(self):
        """Test installing plugin from local directory."""
        with TemporaryDirectory() as tmpdir:
            # Create a mock plugin directory
            plugin_src = Path(tmpdir) / "source" / "test-plugin"
            plugin_src.mkdir(parents=True)

            # Create plugin.json
            plugin_json = plugin_src / "plugin.json"
            plugin_json.write_text(json.dumps({
                "id": "test-plugin",
                "name": "Test Plugin",
                "version": "1.0.0",
            }))

            # Create __init__.py
            init_py = plugin_src / "__init__.py"
            init_py.write_text("# Test plugin")

            # Install
            plugins_dir = Path(tmpdir) / "plugins"
            plugins_dir.mkdir()
            config = SandboxConfig(plugins_dir=str(plugins_dir))
            manager = PluginManager(config)

            plugin = await manager.install_from_directory(plugin_src)

            assert plugin.metadata.id == "test-plugin"
            assert plugin.metadata.name == "Test Plugin"
            assert plugin.status == PluginStatus.INSTALLED

    @pytest.mark.asyncio
    async def test_install_from_nonexistent_directory(self):
        """Test installing from non-existent directory raises error."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            with pytest.raises(PluginInstallError):
                await manager.install_from_directory("/nonexistent/path")

    def test_configure_plugin(self):
        """Test configuring a plugin."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            # Add a plugin directly
            metadata = PluginMetadata(
                id="test-plugin",
                name="Test Plugin",
                version="1.0.0",
            )
            plugin = Plugin(metadata=metadata)
            manager._plugins["test-plugin"] = plugin

            # Configure it
            updated = manager.configure_plugin("test-plugin", {"key": "value"})

            assert updated is not None
            assert updated.config["key"] == "value"

    def test_configure_nonexistent_plugin(self):
        """Test configuring non-existent plugin returns None."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            result = manager.configure_plugin("nonexistent", {"key": "value"})
            assert result is None

    def test_uninstall_plugin(self):
        """Test uninstalling a plugin."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            # Add a plugin directly
            metadata = PluginMetadata(
                id="test-plugin",
                name="Test Plugin",
                version="1.0.0",
            )
            plugin = Plugin(metadata=metadata)
            manager._plugins["test-plugin"] = plugin

            # Uninstall
            result = manager.uninstall("test-plugin")

            assert result is True
            assert "test-plugin" not in manager._plugins

    def test_uninstall_nonexistent_plugin(self):
        """Test uninstalling non-existent plugin returns False."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            result = manager.uninstall("nonexistent")
            assert result is False

    def test_disable_plugin(self):
        """Test disabling a plugin."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            # Add an enabled plugin
            metadata = PluginMetadata(
                id="test-plugin",
                name="Test Plugin",
                version="1.0.0",
            )
            plugin = Plugin(metadata=metadata, status=PluginStatus.ENABLED)
            manager._plugins["test-plugin"] = plugin

            # Disable
            disabled = manager.disable("test-plugin")

            assert disabled is not None
            assert disabled.status == PluginStatus.DISABLED

    def test_list_plugins_with_filter(self):
        """Test listing plugins with status filter."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            # Add plugins with different statuses
            for i, status in enumerate([PluginStatus.ENABLED, PluginStatus.DISABLED]):
                metadata = PluginMetadata(
                    id=f"plugin-{i}",
                    name=f"Plugin {i}",
                    version="1.0.0",
                )
                plugin = Plugin(metadata=metadata, status=status)
                manager._plugins[f"plugin-{i}"] = plugin

            enabled = manager.list_plugins(status=PluginStatus.ENABLED)
            assert len(enabled) == 1
            assert enabled[0].metadata.id == "plugin-0"

            disabled = manager.list_plugins(status=PluginStatus.DISABLED)
            assert len(disabled) == 1
            assert disabled[0].metadata.id == "plugin-1"


class TestCodeExecutor:
    """Tests for CodeExecutor."""

    def test_create_executor(self):
        """Test creating a code executor."""
        config = SandboxConfig()
        executor = CodeExecutor(config)

        assert executor.config == config
        # Executor uses ThreadPoolExecutor for async execution
        assert executor._executor is not None

    def test_get_executor_singleton(self):
        """Test singleton pattern for executor."""
        executor1 = get_executor()
        executor2 = get_executor()

        assert executor1 is executor2

    @pytest.mark.asyncio
    async def test_execute_blocked_code(self):
        """Test executing code with blocked patterns."""
        config = SandboxConfig()
        executor = CodeExecutor(config)

        result = await executor.execute(
            language=CodeLanguage.PYTHON3,
            code="import os\nos.system('ls')",
            inputs={},
        )

        assert result.success is False
        assert "blocked" in result.error.lower() or "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """Test execution timeout."""
        config = SandboxConfig(execution_timeout=1)
        executor = CodeExecutor(config)

        # This would timeout if actually executed
        result = await executor.execute(
            language=CodeLanguage.PYTHON3,
            code="while True: pass",
            inputs={},
            timeout=1,
        )

        # Either blocked or timed out
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_jinja2_simple(self):
        """Test simple Jinja2 template execution."""
        pytest.importorskip("jinja2")  # Skip if jinja2 not installed
        config = SandboxConfig()
        executor = CodeExecutor(config)

        result = await executor.execute(
            language=CodeLanguage.JINJA2,
            code="Hello, {{ name }}!",
            inputs={"name": "World"},
        )

        assert result.success is True
        assert result.output == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_jinja2_with_loop(self):
        """Test Jinja2 template with loop."""
        pytest.importorskip("jinja2")  # Skip if jinja2 not installed
        config = SandboxConfig()
        executor = CodeExecutor(config)

        result = await executor.execute(
            language=CodeLanguage.JINJA2,
            code="{% for i in items %}{{ i }}{% endfor %}",
            inputs={"items": ["a", "b", "c"]},
        )

        assert result.success is True
        assert result.output == "abc"


class TestExecutionError:
    """Tests for ExecutionError."""

    def test_create_execution_error(self):
        """Test creating an execution error."""
        error = ExecutionError("Test error", "TestType")

        assert str(error) == "Test error"
        assert error.error_type == "TestType"

    def test_execution_error_default_type(self):
        """Test default error type."""
        error = ExecutionError("Test error")

        assert error.error_type == "ExecutionError"


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_valid_result(self):
        """Test valid result."""
        result = ValidationResult(valid=True, value=42)

        assert result.valid is True
        assert result.value == 42
        assert result.error is None

    def test_invalid_result(self):
        """Test invalid result."""
        result = ValidationResult(valid=False, error="Validation failed")

        assert result.valid is False
        assert result.value is None
        assert result.error == "Validation failed"


class TestSandboxIntegration:
    """Integration tests for sandbox system."""

    @pytest.mark.asyncio
    async def test_jinja2_complex_template(self):
        """Test complex Jinja2 template."""
        pytest.importorskip("jinja2")  # Skip if jinja2 not installed
        config = SandboxConfig()
        executor = CodeExecutor(config)

        template = """
        {% for user in users %}
        Name: {{ user.name }}, Age: {{ user.age }}
        {% endfor %}
        """
        inputs = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
        }

        result = await executor.execute(
            language=CodeLanguage.JINJA2,
            code=template,
            inputs=inputs,
        )

        assert result.success is True
        assert "Alice" in result.output
        assert "Bob" in result.output

    @pytest.mark.asyncio
    async def test_jinja2_conditionals(self):
        """Test Jinja2 conditionals."""
        pytest.importorskip("jinja2")  # Skip if jinja2 not installed
        config = SandboxConfig()
        executor = CodeExecutor(config)

        template = "{% if show %}visible{% else %}hidden{% endif %}"

        result1 = await executor.execute(
            language=CodeLanguage.JINJA2,
            code=template,
            inputs={"show": True},
        )
        assert result1.success is True
        assert result1.output == "visible"

        result2 = await executor.execute(
            language=CodeLanguage.JINJA2,
            code=template,
            inputs={"show": False},
        )
        assert result2.success is True
        assert result2.output == "hidden"

    def test_validator_with_complex_structure(self):
        """Test validator with complex nested structure."""
        validator = OutputValidator()

        data = {
            "users": [
                {"name": "Alice", "scores": [95, 87, 92]},
                {"name": "Bob", "scores": [88, 91, 85]},
            ],
            "metadata": {
                "total": 2,
                "average": 89.67,
            },
        }

        result = validator.validate(data)

        assert result.valid is True
        assert result.value == data

    def test_plugin_manifest_persistence(self):
        """Test plugin manifest is persisted."""
        with TemporaryDirectory() as tmpdir:
            config = SandboxConfig(plugins_dir=tmpdir)
            manager = PluginManager(config)

            # Add a plugin
            metadata = PluginMetadata(
                id="test-plugin",
                name="Test Plugin",
                version="1.0.0",
            )
            plugin = Plugin(metadata=metadata)
            manager._plugins["test-plugin"] = plugin
            manager._save_manifest()

            # Check manifest exists
            manifest_path = Path(tmpdir) / "manifest.json"
            assert manifest_path.exists()

            # Load manifest
            with open(manifest_path) as f:
                manifest = json.load(f)

            assert len(manifest["plugins"]) == 1
            assert manifest["plugins"][0]["metadata"]["id"] == "test-plugin"


class TestSandboxClient:
    """Tests for SandboxClient."""

    def test_create_client(self):
        """Test creating a sandbox client."""
        from app.core.sandbox.client import SandboxClient

        config = SandboxConfig(remote_endpoint="http://localhost:8194")
        client = SandboxClient(config)

        assert client.config.remote_endpoint == "http://localhost:8194"

    def test_sandbox_response_dataclass(self):
        """Test SandboxResponse dataclass."""
        from app.core.sandbox.client import SandboxResponse

        response = SandboxResponse(
            success=True,
            output={"key": "value"},
            stdout="output",
        )

        assert response.success is True
        assert response.output == {"key": "value"}
        assert response.stdout == "output"
        assert response.error is None

    def test_sandbox_response_error(self):
        """Test SandboxResponse with error."""
        from app.core.sandbox.client import SandboxResponse

        response = SandboxResponse(
            success=False,
            error="Connection failed",
            error_type="ConnectionError",
        )

        assert response.success is False
        assert response.error == "Connection failed"
        assert response.error_type == "ConnectionError"

    def test_extract_result(self):
        """Test result extraction from output."""
        from app.core.sandbox.client import SandboxClient

        client = SandboxClient()

        # Test JSON result
        output = '<<RESULT>>{"key": "value"}<<RESULT>>'
        result = client._extract_result(output)
        assert result == {"key": "value"}

        # Test string result
        output = '<<RESULT>>"hello"<<RESULT>>'
        result = client._extract_result(output)
        assert result == "hello"

        # Test no result
        output = "no markers"
        result = client._extract_result(output)
        assert result is None

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self):
        """Test health check returns False on connection error."""
        from app.core.sandbox.client import SandboxClient

        config = SandboxConfig(remote_endpoint="http://nonexistent:9999")
        client = SandboxClient(config)

        is_healthy = await client.health_check()
        assert is_healthy is False

        await client.close()

    @pytest.mark.asyncio
    async def test_execute_connection_error(self):
        """Test execute returns error on connection failure."""
        from app.core.sandbox.client import SandboxClient

        config = SandboxConfig(remote_endpoint="http://nonexistent:9999")
        client = SandboxClient(config)

        response = await client.execute(
            language="python3",
            code="print('hello')",
        )

        assert response.success is False
        # Connection errors can manifest as various error types
        assert response.error_type in ("ConnectionError", "ConnectError", "JSONDecodeError")
        assert response.error is not None

        await client.close()

    def test_get_sandbox_client_singleton(self):
        """Test singleton pattern for client."""
        from app.core.sandbox.client import get_sandbox_client

        client1 = get_sandbox_client()
        client2 = get_sandbox_client()

        assert client1 is client2
