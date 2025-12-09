"""
Tool/Function calling system for LLM agents.
Provides OpenAI-compatible function definitions and execution.
"""
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, get_type_hints


class ParameterType(str, Enum):
    """JSON Schema parameter types."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    param_type: ParameterType
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[str] | None = None
    items: dict[str, Any] | None = None  # For array types

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.param_type.value,
        }
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.items and self.param_type == ParameterType.ARRAY:
            schema["items"] = self.items
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Definition of a tool for LLM function calling."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    strict: bool = False  # OpenAI strict mode

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function definition format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                "strict": self.strict,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai_format(cls, data: dict[str, Any]) -> "ToolCall":
        """Create from OpenAI tool call format."""
        function = data.get("function", {})
        args_str = function.get("arguments", "{}")

        # Parse arguments JSON
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = {}

        return cls(
            id=data.get("id", ""),
            name=function.get("name", ""),
            arguments=arguments,
        )


@dataclass
class ToolResult:
    """Result of executing a tool."""

    call_id: str
    name: str
    result: Any
    error: str | None = None
    success: bool = True

    def to_openai_message(self) -> dict[str, Any]:
        """Convert to OpenAI tool result message format."""
        if self.success:
            content = json.dumps(self.result) if not isinstance(self.result, str) else self.result
        else:
            content = json.dumps({"error": self.error})

        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": content,
        }


class Tool(ABC):
    """Base class for executable tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Get the tool definition."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        pass


class FunctionTool(Tool):
    """Tool that wraps a Python function."""

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ):
        """
        Create a tool from a function.

        Args:
            func: The function to wrap.
            name: Tool name (defaults to function name).
            description: Tool description (defaults to docstring).
        """
        self.func = func
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""

        super().__init__(tool_name, tool_description)

        # Build parameters from function signature
        self._parameters = self._build_parameters()

    def _python_type_to_param_type(self, python_type: type) -> ParameterType:
        """Map Python types to parameter types."""
        type_map = {
            str: ParameterType.STRING,
            int: ParameterType.INTEGER,
            float: ParameterType.NUMBER,
            bool: ParameterType.BOOLEAN,
            list: ParameterType.ARRAY,
            dict: ParameterType.OBJECT,
        }
        return type_map.get(python_type, ParameterType.STRING)

    def _build_parameters(self) -> list[ToolParameter]:
        """Build parameters from function signature."""
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func) if hasattr(self.func, "__annotations__") else {}

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            python_type = hints.get(param_name, str)
            param_type = self._python_type_to_param_type(python_type)

            has_default = param.default != inspect.Parameter.empty
            default_value = param.default if has_default else None

            parameters.append(
                ToolParameter(
                    name=param_name,
                    param_type=param_type,
                    required=not has_default,
                    default=default_value,
                )
            )

        return parameters

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self._parameters,
        )

    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped function."""
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a function as a tool."""
        tool = FunctionTool(func, name, description)
        self.register(tool)

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> list[ToolDefinition]:
        """Get definitions for all registered tools."""
        return [tool.get_definition() for tool in self._tools.values()]

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI format."""
        return [tool.get_definition().to_openai_format() for tool in self._tools.values()]

    def to_anthropic_format(self) -> list[dict[str, Any]]:
        """Get all tools in Anthropic format."""
        return [tool.get_definition().to_anthropic_format() for tool in self._tools.values()]

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


class ToolExecutor:
    """Executes tool calls from LLM responses."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            tool_call: The tool call to execute.

        Returns:
            Result of the tool execution.
        """
        tool = self.registry.get(tool_call.name)

        if tool is None:
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=f"Unknown tool: {tool_call.name}",
                success=False,
            )

        try:
            result = await tool.execute(**tool_call.arguments)
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                result=result,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e),
                success=False,
            )

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of tool calls to execute.

        Returns:
            List of results.
        """
        results = []
        for tool_call in tool_calls:
            result = await self.execute(tool_call)
            results.append(result)
        return results

    def parse_openai_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse tool calls from OpenAI format."""
        return [ToolCall.from_openai_format(tc) for tc in tool_calls]


def tool(
    name: str | None = None,
    description: str | None = None,
):
    """
    Decorator to register a function as a tool.

    Usage:
        @tool(name="my_tool", description="Does something")
        def my_function(arg1: str, arg2: int = 10):
            ...
    """

    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(func, name, description)

    return decorator


# Pre-built tools
class GetCurrentTimeTool(Tool):
    """Tool to get the current time."""

    def __init__(self):
        super().__init__(
            name="get_current_time",
            description="Get the current date and time",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="timezone",
                    param_type=ParameterType.STRING,
                    description="Timezone (e.g., 'UTC', 'America/New_York')",
                    required=False,
                    default="UTC",
                ),
            ],
        )

    async def execute(self, timezone: str = "UTC") -> dict[str, str]:
        import zoneinfo
        from datetime import datetime

        try:
            tz = zoneinfo.ZoneInfo(timezone)
            now = datetime.now(tz)
        except Exception:
            from datetime import timezone as tz_module
            now = datetime.now(tz_module.utc)

        return {
            "datetime": now.isoformat(),
            "timezone": timezone,
        }


class CalculatorTool(Tool):
    """Simple calculator tool."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform basic math calculations",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="expression",
                    param_type=ParameterType.STRING,
                    description="Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
                    required=True,
                ),
            ],
        )

    async def execute(self, expression: str) -> dict[str, Any]:
        # Safe evaluation of mathematical expressions
        allowed_chars = set("0123456789+-*/().% ")

        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}

        try:
            result = eval(expression)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e)}


class JsonParseTool(Tool):
    """Tool to parse and query JSON data."""

    def __init__(self):
        super().__init__(
            name="json_parse",
            description="Parse JSON string and optionally extract a value by path",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="json_string",
                    param_type=ParameterType.STRING,
                    description="JSON string to parse",
                    required=True,
                ),
                ToolParameter(
                    name="path",
                    param_type=ParameterType.STRING,
                    description="Dot-notation path to extract (e.g., 'user.name')",
                    required=False,
                ),
            ],
        )

    async def execute(self, json_string: str, path: str | None = None) -> dict[str, Any]:
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}

        if path:
            parts = path.split(".")
            value = data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                elif isinstance(value, list):
                    try:
                        idx = int(part)
                        value = value[idx]
                    except (ValueError, IndexError):
                        return {"error": f"Path '{path}' not found"}
                else:
                    return {"error": f"Path '{path}' not found"}
            return {"path": path, "value": value}

        return {"data": data}


class JsonFormatTool(Tool):
    """Tool to format data as JSON string."""

    def __init__(self):
        super().__init__(
            name="json_format",
            description="Convert data to formatted JSON string",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="data",
                    param_type=ParameterType.OBJECT,
                    description="Data object to convert to JSON",
                    required=True,
                ),
                ToolParameter(
                    name="indent",
                    param_type=ParameterType.INTEGER,
                    description="Indentation level (0 for compact)",
                    required=False,
                    default=2,
                ),
            ],
        )

    async def execute(self, data: Any, indent: int = 2) -> dict[str, Any]:
        try:
            indent_val = indent if indent > 0 else None
            formatted = json.dumps(data, indent=indent_val, ensure_ascii=False)
            return {"json": formatted}
        except (TypeError, ValueError) as e:
            return {"error": f"Cannot serialize to JSON: {e}"}


class StringFormatTool(Tool):
    """Tool to format strings with variables."""

    def __init__(self):
        super().__init__(
            name="string_format",
            description="Format a template string with variables",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="template",
                    param_type=ParameterType.STRING,
                    description="Template string with {placeholders}",
                    required=True,
                ),
                ToolParameter(
                    name="variables",
                    param_type=ParameterType.OBJECT,
                    description="Variables to substitute",
                    required=True,
                ),
            ],
        )

    async def execute(self, template: str, variables: dict[str, Any]) -> dict[str, Any]:
        try:
            result = template.format(**variables)
            return {"result": result}
        except KeyError as e:
            return {"error": f"Missing variable: {e}"}
        except Exception as e:
            return {"error": str(e)}


class StringSearchTool(Tool):
    """Tool to search within strings."""

    def __init__(self):
        super().__init__(
            name="string_search",
            description="Search for pattern in text, supports regex",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="text",
                    param_type=ParameterType.STRING,
                    description="Text to search in",
                    required=True,
                ),
                ToolParameter(
                    name="pattern",
                    param_type=ParameterType.STRING,
                    description="Pattern to search for (regex supported)",
                    required=True,
                ),
                ToolParameter(
                    name="regex",
                    param_type=ParameterType.BOOLEAN,
                    description="Whether to use regex matching",
                    required=False,
                    default=False,
                ),
            ],
        )

    async def execute(
        self, text: str, pattern: str, regex: bool = False
    ) -> dict[str, Any]:
        import re

        if regex:
            try:
                matches = re.findall(pattern, text)
                return {"matches": matches, "count": len(matches)}
            except re.error as e:
                return {"error": f"Invalid regex: {e}"}
        else:
            count = text.count(pattern)
            positions = []
            start = 0
            while True:
                pos = text.find(pattern, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            return {"count": count, "positions": positions}


class RandomTool(Tool):
    """Tool to generate random values."""

    def __init__(self):
        super().__init__(
            name="random",
            description="Generate random numbers or UUIDs",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="type",
                    param_type=ParameterType.STRING,
                    description="Type of random value",
                    required=True,
                    enum=["integer", "float", "uuid", "choice"],
                ),
                ToolParameter(
                    name="min",
                    param_type=ParameterType.NUMBER,
                    description="Minimum value (for integer/float)",
                    required=False,
                    default=0,
                ),
                ToolParameter(
                    name="max",
                    param_type=ParameterType.NUMBER,
                    description="Maximum value (for integer/float)",
                    required=False,
                    default=100,
                ),
                ToolParameter(
                    name="choices",
                    param_type=ParameterType.ARRAY,
                    description="List of choices (for choice type)",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        type: str,
        min: float = 0,
        max: float = 100,
        choices: list[Any] | None = None,
    ) -> dict[str, Any]:
        import random
        import uuid

        if type == "integer":
            value = random.randint(int(min), int(max))
            return {"value": value, "type": "integer"}
        elif type == "float":
            value = random.uniform(min, max)
            return {"value": value, "type": "float"}
        elif type == "uuid":
            value = str(uuid.uuid4())
            return {"value": value, "type": "uuid"}
        elif type == "choice":
            if not choices:
                return {"error": "No choices provided"}
            value = random.choice(choices)
            return {"value": value, "type": "choice"}
        else:
            return {"error": f"Unknown type: {type}"}


class HttpRequestTool(Tool):
    """Tool to make HTTP requests."""

    def __init__(self):
        super().__init__(
            name="http_request",
            description="Make HTTP GET or POST requests",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="url",
                    param_type=ParameterType.STRING,
                    description="URL to request",
                    required=True,
                ),
                ToolParameter(
                    name="method",
                    param_type=ParameterType.STRING,
                    description="HTTP method",
                    required=False,
                    default="GET",
                    enum=["GET", "POST", "PUT", "DELETE", "PATCH"],
                ),
                ToolParameter(
                    name="headers",
                    param_type=ParameterType.OBJECT,
                    description="Request headers",
                    required=False,
                ),
                ToolParameter(
                    name="body",
                    param_type=ParameterType.OBJECT,
                    description="Request body (for POST/PUT/PATCH)",
                    required=False,
                ),
                ToolParameter(
                    name="timeout",
                    param_type=ParameterType.INTEGER,
                    description="Request timeout in seconds",
                    required=False,
                    default=30,
                ),
            ],
        )

    async def execute(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method=method.upper(),
                    url=url,
                    headers=headers,
                    json=body if body else None,
                )
                try:
                    data = response.json()
                except Exception:
                    data = response.text

                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "data": data,
                }
        except httpx.TimeoutException:
            return {"error": "Request timed out"}
        except httpx.RequestError as e:
            return {"error": f"Request failed: {e}"}


class Base64Tool(Tool):
    """Tool to encode/decode base64."""

    def __init__(self):
        super().__init__(
            name="base64",
            description="Encode or decode base64 strings",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    param_type=ParameterType.STRING,
                    description="Action to perform",
                    required=True,
                    enum=["encode", "decode"],
                ),
                ToolParameter(
                    name="data",
                    param_type=ParameterType.STRING,
                    description="String to encode or decode",
                    required=True,
                ),
            ],
        )

    async def execute(self, action: str, data: str) -> dict[str, Any]:
        import base64

        if action == "encode":
            result = base64.b64encode(data.encode()).decode()
            return {"result": result}
        elif action == "decode":
            try:
                result = base64.b64decode(data).decode()
                return {"result": result}
            except Exception as e:
                return {"error": f"Decode failed: {e}"}
        else:
            return {"error": f"Unknown action: {action}"}


class HashTool(Tool):
    """Tool to compute hashes."""

    def __init__(self):
        super().__init__(
            name="hash",
            description="Compute hash of a string",
        )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="data",
                    param_type=ParameterType.STRING,
                    description="String to hash",
                    required=True,
                ),
                ToolParameter(
                    name="algorithm",
                    param_type=ParameterType.STRING,
                    description="Hash algorithm",
                    required=False,
                    default="sha256",
                    enum=["md5", "sha1", "sha256", "sha512"],
                ),
            ],
        )

    async def execute(self, data: str, algorithm: str = "sha256") -> dict[str, Any]:
        import hashlib

        algo_map = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
        }
        hash_func = algo_map.get(algorithm)
        if not hash_func:
            return {"error": f"Unknown algorithm: {algorithm}"}

        hash_value = hash_func(data.encode()).hexdigest()
        return {"hash": hash_value, "algorithm": algorithm}


def get_default_tools() -> list[Tool]:
    """Get all default builtin tools."""
    return [
        GetCurrentTimeTool(),
        CalculatorTool(),
        JsonParseTool(),
        JsonFormatTool(),
        StringFormatTool(),
        StringSearchTool(),
        RandomTool(),
        HttpRequestTool(),
        Base64Tool(),
        HashTool(),
    ]


def create_default_registry() -> ToolRegistry:
    """Create a registry with all default tools."""
    registry = ToolRegistry()
    for tool in get_default_tools():
        registry.register(tool)
    return registry
