"""
Unit tests for prompt template system.
Tests template rendering, validation, and versioning.
"""
import pytest
from datetime import datetime

from app.core.prompts import (
    TemplateFormat,
    PromptVariable,
    PromptTemplate,
    PromptBuilder,
    PromptRegistry,
    create_prompt_template,
    get_system_prompt,
    SYSTEM_PROMPTS,
)


class TestTemplateFormat:
    """Tests for template format enum."""

    def test_format_values(self):
        """Test format enum values."""
        assert TemplateFormat.PYTHON.value == "python"
        assert TemplateFormat.JINJA2.value == "jinja2"
        assert TemplateFormat.FSTRING.value == "fstring"


class TestPromptVariable:
    """Tests for prompt variables."""

    def test_create_basic_variable(self):
        """Test creating a basic variable."""
        var = PromptVariable(name="name")

        assert var.name == "name"
        assert var.required is True
        assert var.default is None

    def test_variable_with_default(self):
        """Test variable with default value."""
        var = PromptVariable(
            name="limit",
            default=10,
            required=False,
        )

        assert var.default == 10
        assert var.required is False

    def test_variable_with_enum(self):
        """Test variable with enum constraint."""
        var = PromptVariable(
            name="format",
            enum=["json", "xml"],
        )

        assert var.enum == ["json", "xml"]

    def test_validate_required_missing(self):
        """Test validation fails for missing required variable."""
        var = PromptVariable(name="query", required=True)

        is_valid, error = var.validate(None)

        assert is_valid is False
        assert "Required variable" in error

    def test_validate_required_present(self):
        """Test validation passes for present required variable."""
        var = PromptVariable(name="query", required=True)

        is_valid, error = var.validate("test")

        assert is_valid is True
        assert error is None

    def test_validate_enum_valid(self):
        """Test validation passes for valid enum value."""
        var = PromptVariable(name="format", enum=["json", "xml"])

        is_valid, error = var.validate("json")

        assert is_valid is True

    def test_validate_enum_invalid(self):
        """Test validation fails for invalid enum value."""
        var = PromptVariable(name="format", enum=["json", "xml"])

        is_valid, error = var.validate("csv")

        assert is_valid is False
        assert "not in allowed values" in error


class TestPromptTemplate:
    """Tests for prompt templates."""

    def test_create_basic_template(self):
        """Test creating a basic template."""
        template = PromptTemplate(
            name="greeting",
            template="Hello, {name}!",
        )

        assert template.name == "greeting"
        assert template.version == "1.0.0"

    def test_template_with_variables(self):
        """Test template with defined variables."""
        template = PromptTemplate(
            name="greeting",
            template="Hello, {name}!",
            variables=[
                PromptVariable(name="name", description="User's name"),
            ],
        )

        assert len(template.variables) == 1

    def test_render_simple_template(self):
        """Test rendering a simple template."""
        template = PromptTemplate(
            name="greeting",
            template="Hello, {name}!",
        )

        result = template.render(name="Alice")

        assert result == "Hello, Alice!"

    def test_render_with_defaults(self):
        """Test rendering with default values."""
        template = PromptTemplate(
            name="greeting",
            template="Hello, {name}! You have {count} messages.",
            variables=[
                PromptVariable(name="name"),
                PromptVariable(name="count", default=0, required=False),
            ],
        )

        result = template.render(name="Bob")

        assert "Bob" in result
        assert "0" in result

    def test_render_missing_required(self):
        """Test rendering fails with missing required variable."""
        template = PromptTemplate(
            name="greeting",
            template="Hello, {name}!",
            variables=[
                PromptVariable(name="name", required=True),
            ],
        )

        with pytest.raises(ValueError) as exc:
            template.render()

        assert "Required variable" in str(exc.value)

    def test_callable_template(self):
        """Test calling template directly."""
        template = PromptTemplate(
            name="test",
            template="Value: {x}",
        )

        result = template(x=42)

        assert result == "Value: 42"

    def test_extract_variables(self):
        """Test extracting variables from template."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}, your age is {age}.",
        )

        var_names = template.get_variable_names()

        assert "name" in var_names
        assert "age" in var_names

    def test_jinja2_format(self):
        """Test Jinja2-style template."""
        template = PromptTemplate(
            name="test",
            template="Hello {{ name }}!",
            format=TemplateFormat.JINJA2,
        )

        result = template.render(name="World")

        assert result == "Hello World!"

    def test_template_metadata(self):
        """Test template metadata."""
        template = PromptTemplate(
            name="test",
            template="Test",
            tags=["assistant", "general"],
            metadata={"author": "test"},
        )

        assert "assistant" in template.tags
        assert template.metadata["author"] == "test"

    def test_template_has_timestamp(self):
        """Test template has creation timestamp."""
        template = PromptTemplate(name="test", template="Test")

        assert template.created_at is not None
        assert isinstance(template.created_at, datetime)

    def test_validate_variables(self):
        """Test variable validation."""
        template = PromptTemplate(
            name="test",
            template="{query}",
            variables=[
                PromptVariable(name="query", required=True),
            ],
        )

        errors = template.validate_variables({})

        assert len(errors) > 0

    def test_render_multiple_variables(self):
        """Test rendering with multiple variables."""
        template = PromptTemplate(
            name="context",
            template="User: {user}\nTask: {task}\nContext: {context}",
        )

        result = template.render(
            user="Alice",
            task="summarize",
            context="Some data",
        )

        assert "Alice" in result
        assert "summarize" in result
        assert "Some data" in result


class TestPromptBuilder:
    """Tests for prompt builder."""

    def test_create_builder(self):
        """Test creating a builder."""
        builder = PromptBuilder()

        assert builder is not None

    def test_add_text(self):
        """Test adding text."""
        builder = PromptBuilder()
        builder.add("Hello")
        builder.add(" World")

        result = builder.build()

        assert result == "Hello World"

    def test_add_line(self):
        """Test adding line with newline."""
        builder = PromptBuilder()
        builder.add_line("Line 1")
        builder.add_line("Line 2")

        result = builder.build()

        assert "Line 1\n" in result
        assert "Line 2\n" in result

    def test_add_section(self):
        """Test adding sections."""
        builder = PromptBuilder()
        builder.add_section("Context", "Some context here")

        result = builder.build()

        assert "## Context" in result
        assert "Some context here" in result

    def test_add_if_true(self):
        """Test conditional add when true."""
        builder = PromptBuilder()
        builder.add_if(True, "Included")

        result = builder.build()

        assert "Included" in result

    def test_add_if_false(self):
        """Test conditional add when false."""
        builder = PromptBuilder()
        builder.add_if(False, "Not Included")

        result = builder.build()

        assert "Not Included" not in result

    def test_set_variable(self):
        """Test setting variables."""
        builder = PromptBuilder()
        builder.add("Hello {name}")
        builder.set_variable("name", "World")

        result = builder.build()

        assert result == "Hello World"

    def test_method_chaining(self):
        """Test builder supports method chaining."""
        result = (
            PromptBuilder()
            .add("Hello ")
            .add("World")
            .build()
        )

        assert result == "Hello World"

    def test_clear(self):
        """Test clearing the builder."""
        builder = PromptBuilder()
        builder.add("Something")
        builder.clear()

        result = builder.build()

        assert result == ""


class TestPromptRegistry:
    """Tests for prompt registry."""

    def test_create_registry(self):
        """Test creating a registry."""
        registry = PromptRegistry()

        assert len(registry) == 0

    def test_register_template(self):
        """Test registering a template."""
        registry = PromptRegistry()
        template = PromptTemplate(name="test", template="Test")

        registry.register(template)

        assert "test" in registry
        assert len(registry) == 1

    def test_get_template(self):
        """Test getting a template."""
        registry = PromptRegistry()
        template = PromptTemplate(name="greeting", template="Hello!")
        registry.register(template)

        retrieved = registry.get("greeting")

        assert retrieved is not None
        assert retrieved.template == "Hello!"

    def test_get_nonexistent(self):
        """Test getting nonexistent template."""
        registry = PromptRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_multiple_versions(self):
        """Test registering multiple versions."""
        registry = PromptRegistry()

        v1 = PromptTemplate(name="test", template="V1", version="1.0.0")
        v2 = PromptTemplate(name="test", template="V2", version="2.0.0")

        registry.register(v1)
        registry.register(v2)

        # Default should return latest
        latest = registry.get("test")
        assert latest.template == "V2"

        # Can get specific version
        specific = registry.get("test", version="1.0.0")
        assert specific.template == "V1"

    def test_get_all_versions(self):
        """Test getting all versions."""
        registry = PromptRegistry()

        v1 = PromptTemplate(name="test", template="V1", version="1.0.0")
        v2 = PromptTemplate(name="test", template="V2", version="2.0.0")
        registry.register(v1)
        registry.register(v2)

        versions = registry.get_all_versions("test")

        assert len(versions) == 2

    def test_list_templates(self):
        """Test listing all template names."""
        registry = PromptRegistry()
        registry.register(PromptTemplate(name="a", template="A"))
        registry.register(PromptTemplate(name="b", template="B"))

        names = registry.list_templates()

        assert "a" in names
        assert "b" in names

    def test_unregister_template(self):
        """Test unregistering a template."""
        registry = PromptRegistry()
        registry.register(PromptTemplate(name="test", template="Test"))
        registry.unregister("test")

        assert "test" not in registry

    def test_unregister_specific_version(self):
        """Test unregistering specific version."""
        registry = PromptRegistry()
        registry.register(PromptTemplate(name="test", template="V1", version="1.0.0"))
        registry.register(PromptTemplate(name="test", template="V2", version="2.0.0"))

        registry.unregister("test", version="1.0.0")

        versions = registry.get_all_versions("test")
        assert len(versions) == 1
        assert versions[0].version == "2.0.0"


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_prompt_template(self):
        """Test creating template with factory."""
        template = create_prompt_template(
            name="greeting",
            template="Hello, {name}!",
            description="A greeting template",
            variables=[
                {"name": "name", "description": "Person's name"},
            ],
        )

        assert template.name == "greeting"
        assert len(template.variables) == 1

    def test_create_template_without_variables(self):
        """Test creating template without variables."""
        template = create_prompt_template(
            name="static",
            template="This is static text",
        )

        assert template.name == "static"
        assert len(template.variables) == 0


class TestSystemPrompts:
    """Tests for pre-built system prompts."""

    def test_system_prompts_exist(self):
        """Test pre-built prompts exist."""
        assert "helpful_assistant" in SYSTEM_PROMPTS
        assert "code_assistant" in SYSTEM_PROMPTS
        assert "qa_assistant" in SYSTEM_PROMPTS

    def test_get_system_prompt(self):
        """Test getting system prompt by name."""
        prompt = get_system_prompt("helpful_assistant")

        assert prompt is not None
        assert "helpful" in prompt.template.lower()

    def test_code_assistant_prompt(self):
        """Test code assistant can be rendered."""
        prompt = get_system_prompt("code_assistant")

        result = prompt.render(context="Python development")

        assert "Python development" in result

    def test_qa_assistant_prompt(self):
        """Test Q&A assistant requires context."""
        prompt = get_system_prompt("qa_assistant")

        result = prompt.render(context="Some retrieved documents")

        assert "Some retrieved documents" in result

    def test_qa_assistant_missing_context(self):
        """Test Q&A assistant fails without context."""
        prompt = get_system_prompt("qa_assistant")

        with pytest.raises(ValueError):
            prompt.render()  # Missing required context

    def test_get_nonexistent_system_prompt(self):
        """Test getting nonexistent system prompt."""
        prompt = get_system_prompt("nonexistent")

        assert prompt is None
