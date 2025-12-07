"""
Unit tests for tool/function calling system.
Tests tool definitions, execution, and OpenAI compatibility.
"""
import pytest
from unittest.mock import Mock, AsyncMock

from app.core.tools import (
    ParameterType,
    ToolParameter,
    ToolDefinition,
    ToolCall,
    ToolResult,
    Tool,
    FunctionTool,
    ToolRegistry,
    ToolExecutor,
    tool,
    GetCurrentTimeTool,
    CalculatorTool,
)


class TestParameterType:
    """Tests for parameter type enum."""

    def test_parameter_types(self):
        """Test parameter type values."""
        assert ParameterType.STRING.value == "string"
        assert ParameterType.NUMBER.value == "number"
        assert ParameterType.INTEGER.value == "integer"
        assert ParameterType.BOOLEAN.value == "boolean"
        assert ParameterType.ARRAY.value == "array"
        assert ParameterType.OBJECT.value == "object"


class TestToolParameter:
    """Tests for tool parameter definition."""

    def test_create_basic_parameter(self):
        """Test creating a basic parameter."""
        param = ToolParameter(
            name="query",
            param_type=ParameterType.STRING,
        )

        assert param.name == "query"
        assert param.param_type == ParameterType.STRING
        assert param.required is True

    def test_parameter_with_description(self):
        """Test parameter with description."""
        param = ToolParameter(
            name="query",
            param_type=ParameterType.STRING,
            description="The search query",
        )

        assert param.description == "The search query"

    def test_optional_parameter(self):
        """Test optional parameter with default."""
        param = ToolParameter(
            name="limit",
            param_type=ParameterType.INTEGER,
            required=False,
            default=10,
        )

        assert param.required is False
        assert param.default == 10

    def test_enum_parameter(self):
        """Test parameter with enum values."""
        param = ToolParameter(
            name="format",
            param_type=ParameterType.STRING,
            enum=["json", "xml", "csv"],
        )

        assert param.enum == ["json", "xml", "csv"]

    def test_to_schema(self):
        """Test converting to JSON Schema."""
        param = ToolParameter(
            name="query",
            param_type=ParameterType.STRING,
            description="Search query",
        )

        schema = param.to_schema()

        assert schema["type"] == "string"
        assert schema["description"] == "Search query"

    def test_to_schema_with_enum(self):
        """Test schema includes enum."""
        param = ToolParameter(
            name="format",
            param_type=ParameterType.STRING,
            enum=["json", "xml"],
        )

        schema = param.to_schema()

        assert schema["enum"] == ["json", "xml"]


class TestToolDefinition:
    """Tests for tool definition."""

    def test_create_basic_definition(self):
        """Test creating a basic tool definition."""
        tool_def = ToolDefinition(
            name="search",
            description="Search for information",
        )

        assert tool_def.name == "search"
        assert tool_def.description == "Search for information"
        assert len(tool_def.parameters) == 0

    def test_definition_with_parameters(self):
        """Test definition with parameters."""
        tool_def = ToolDefinition(
            name="search",
            description="Search for information",
            parameters=[
                ToolParameter(
                    name="query",
                    param_type=ParameterType.STRING,
                    description="Search query",
                ),
            ],
        )

        assert len(tool_def.parameters) == 1

    def test_to_openai_format(self):
        """Test converting to OpenAI format."""
        tool_def = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters=[
                ToolParameter(
                    name="location",
                    param_type=ParameterType.STRING,
                    description="City name",
                    required=True,
                ),
                ToolParameter(
                    name="unit",
                    param_type=ParameterType.STRING,
                    required=False,
                    default="celsius",
                ),
            ],
        )

        result = tool_def.to_openai_format()

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get weather for a location"
        assert "properties" in result["function"]["parameters"]
        assert "location" in result["function"]["parameters"]["properties"]
        assert "location" in result["function"]["parameters"]["required"]
        assert "unit" not in result["function"]["parameters"]["required"]

    def test_to_anthropic_format(self):
        """Test converting to Anthropic format."""
        tool_def = ToolDefinition(
            name="search",
            description="Search the web",
            parameters=[
                ToolParameter(
                    name="query",
                    param_type=ParameterType.STRING,
                ),
            ],
        )

        result = tool_def.to_anthropic_format()

        assert result["name"] == "search"
        assert result["description"] == "Search the web"
        assert "input_schema" in result
        assert result["input_schema"]["type"] == "object"


class TestToolCall:
    """Tests for tool call parsing."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        call = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "hello"},
        )

        assert call.id == "call_123"
        assert call.name == "search"
        assert call.arguments["query"] == "hello"

    def test_from_openai_format(self):
        """Test parsing from OpenAI format."""
        openai_data = {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris", "unit": "celsius"}',
            },
        }

        call = ToolCall.from_openai_format(openai_data)

        assert call.id == "call_abc123"
        assert call.name == "get_weather"
        assert call.arguments["location"] == "Paris"
        assert call.arguments["unit"] == "celsius"

    def test_from_openai_format_invalid_json(self):
        """Test handling invalid JSON in arguments."""
        openai_data = {
            "id": "call_123",
            "function": {
                "name": "test",
                "arguments": "not valid json",
            },
        }

        call = ToolCall.from_openai_format(openai_data)

        assert call.arguments == {}


class TestToolResult:
    """Tests for tool execution results."""

    def test_create_success_result(self):
        """Test creating a successful result."""
        result = ToolResult(
            call_id="call_123",
            name="search",
            result={"data": "found"},
            success=True,
        )

        assert result.success is True
        assert result.error is None

    def test_create_error_result(self):
        """Test creating an error result."""
        result = ToolResult(
            call_id="call_123",
            name="search",
            result=None,
            error="Tool not found",
            success=False,
        )

        assert result.success is False
        assert result.error == "Tool not found"

    def test_to_openai_message_success(self):
        """Test converting success result to message."""
        result = ToolResult(
            call_id="call_123",
            name="calculator",
            result={"answer": 42},
            success=True,
        )

        message = result.to_openai_message()

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_123"
        assert "42" in message["content"]

    def test_to_openai_message_error(self):
        """Test converting error result to message."""
        result = ToolResult(
            call_id="call_123",
            name="calculator",
            result=None,
            error="Division by zero",
            success=False,
        )

        message = result.to_openai_message()

        assert "error" in message["content"]
        assert "Division by zero" in message["content"]


class TestFunctionTool:
    """Tests for function-based tools."""

    def test_create_from_function(self):
        """Test creating tool from function."""

        def search(query: str, limit: int = 10):
            """Search for something."""
            return f"Found {limit} results for {query}"

        func_tool = FunctionTool(search)

        assert func_tool.name == "search"
        assert "Search for something" in func_tool.description

    def test_function_parameters_inferred(self):
        """Test parameters are inferred from signature."""

        def my_func(name: str, age: int, active: bool = True):
            pass

        func_tool = FunctionTool(my_func)
        params = func_tool._parameters

        assert len(params) == 3
        assert params[0].name == "name"
        assert params[0].param_type == ParameterType.STRING
        assert params[1].name == "age"
        assert params[1].param_type == ParameterType.INTEGER
        assert params[2].name == "active"
        assert params[2].required is False

    def test_custom_name_and_description(self):
        """Test custom name and description."""

        def fn():
            pass

        func_tool = FunctionTool(fn, name="custom_name", description="Custom desc")

        assert func_tool.name == "custom_name"
        assert func_tool.description == "Custom desc"

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Test executing a sync function."""

        def add(a: int, b: int) -> int:
            return a + b

        func_tool = FunctionTool(add)
        result = await func_tool.execute(a=5, b=3)

        assert result == 8

    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        """Test executing an async function."""

        async def async_add(a: int, b: int) -> int:
            return a + b

        func_tool = FunctionTool(async_add)
        result = await func_tool.execute(a=5, b=3)

        assert result == 8


class TestToolRegistry:
    """Tests for tool registry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        func_tool = FunctionTool(lambda x: x, name="test")
        registry.register(func_tool)

        assert "test" in registry
        assert len(registry) == 1

    def test_register_function(self):
        """Test registering a function directly."""
        registry = ToolRegistry()

        def search(query: str):
            return query

        registry.register_function(search)

        assert "search" in registry

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()

        def my_tool():
            pass

        registry.register_function(my_tool)

        tool = registry.get("my_tool")
        assert tool is not None
        assert tool.name == "my_tool"

    def test_get_nonexistent_tool(self):
        """Test getting a nonexistent tool."""
        registry = ToolRegistry()

        tool = registry.get("nonexistent")
        assert tool is None

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()

        def test_fn():
            pass

        registry.register_function(test_fn)
        registry.unregister("test_fn")

        assert "test_fn" not in registry

    def test_get_all_tools(self):
        """Test getting all tools."""
        registry = ToolRegistry()

        registry.register_function(lambda: None, name="tool1")
        registry.register_function(lambda: None, name="tool2")

        tools = registry.get_all()
        assert len(tools) == 2

    def test_to_openai_format(self):
        """Test converting registry to OpenAI format."""
        registry = ToolRegistry()

        def search(query: str):
            """Search for things."""
            pass

        registry.register_function(search)

        openai_tools = registry.to_openai_format()

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "search"


class TestToolExecutor:
    """Tests for tool executor."""

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool call."""
        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        registry.register_function(add)
        executor = ToolExecutor(registry)

        call = ToolCall(id="call_1", name="add", arguments={"a": 5, "b": 3})
        result = await executor.execute(call)

        assert result.success is True
        assert result.result == 8

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing an unknown tool."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        call = ToolCall(id="call_1", name="unknown", arguments={})
        result = await executor.execute(call)

        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        """Test handling tool execution error."""
        registry = ToolRegistry()

        def failing_tool():
            raise ValueError("Something went wrong")

        registry.register_function(failing_tool)
        executor = ToolExecutor(registry)

        call = ToolCall(id="call_1", name="failing_tool", arguments={})
        result = await executor.execute(call)

        assert result.success is False
        assert "Something went wrong" in result.error

    @pytest.mark.asyncio
    async def test_execute_all(self):
        """Test executing multiple tool calls."""
        registry = ToolRegistry()

        def double(n: int) -> int:
            return n * 2

        registry.register_function(double)
        executor = ToolExecutor(registry)

        calls = [
            ToolCall(id="call_1", name="double", arguments={"n": 5}),
            ToolCall(id="call_2", name="double", arguments={"n": 10}),
        ]
        results = await executor.execute_all(calls)

        assert len(results) == 2
        assert results[0].result == 10
        assert results[1].result == 20

    def test_parse_openai_tool_calls(self):
        """Test parsing OpenAI tool calls."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        openai_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "test"}',
                },
            },
        ]

        calls = executor.parse_openai_tool_calls(openai_calls)

        assert len(calls) == 1
        assert calls[0].name == "search"


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_decorator_creates_tool(self):
        """Test decorator creates a FunctionTool."""

        @tool(name="my_tool", description="Does something")
        def my_func(x: int) -> int:
            return x * 2

        assert isinstance(my_func, FunctionTool)
        assert my_func.name == "my_tool"
        assert my_func.description == "Does something"


class TestBuiltInTools:
    """Tests for built-in tools."""

    @pytest.mark.asyncio
    async def test_get_current_time(self):
        """Test get current time tool."""
        time_tool = GetCurrentTimeTool()
        result = await time_tool.execute()

        assert "datetime" in result
        assert "timezone" in result

    @pytest.mark.asyncio
    async def test_get_current_time_with_timezone(self):
        """Test get current time with timezone."""
        time_tool = GetCurrentTimeTool()
        result = await time_tool.execute(timezone="UTC")

        assert result["timezone"] == "UTC"

    def test_time_tool_definition(self):
        """Test time tool definition."""
        time_tool = GetCurrentTimeTool()
        definition = time_tool.get_definition()

        assert definition.name == "get_current_time"
        assert len(definition.parameters) == 1

    @pytest.mark.asyncio
    async def test_calculator_basic(self):
        """Test calculator with basic expression."""
        calc = CalculatorTool()
        result = await calc.execute(expression="2 + 2")

        assert result["result"] == 4

    @pytest.mark.asyncio
    async def test_calculator_complex(self):
        """Test calculator with complex expression."""
        calc = CalculatorTool()
        result = await calc.execute(expression="(10 + 5) * 2")

        assert result["result"] == 30

    @pytest.mark.asyncio
    async def test_calculator_invalid_chars(self):
        """Test calculator rejects invalid characters."""
        calc = CalculatorTool()
        result = await calc.execute(expression="import os")

        assert "error" in result

    def test_calculator_definition(self):
        """Test calculator tool definition."""
        calc = CalculatorTool()
        definition = calc.get_definition()

        assert definition.name == "calculator"
        assert len(definition.parameters) == 1
        assert definition.parameters[0].name == "expression"
