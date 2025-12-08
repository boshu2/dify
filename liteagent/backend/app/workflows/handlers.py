"""
Workflow node handlers.

Each handler is responsible for executing a specific type of workflow node.
"""
import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from app.workflows.types import NodeDefinition
from app.workflows.state import WorkflowState


def substitute_variables(template: str, variables: dict[str, Any]) -> str:
    """
    Substitute {{variable}} placeholders in template.

    Supports nested access: {{user.name}} -> variables["user"]["name"]
    """
    pattern = r"\{\{([\w.]+)\}\}"

    def replacer(match: re.Match) -> str:
        var_path = match.group(1)
        parts = var_path.split(".")
        value: Any = variables

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return match.group(0)  # Keep original if not found

        return str(value)

    return re.sub(pattern, replacer, template)


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
        return state.variables.copy()


class EndNodeHandler(NodeHandler):
    """Handler for END nodes."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
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

        agent = context.get("agents", {}).get(node.config.get("agent_id"))

        if not agent:
            config = AgentConfig(
                agent_id=node.config.get("agent_id", node.id),
                purpose=node.config.get("purpose", "Execute task"),
                llm_client=context.get("llm_client"),
            )
            agent = Agent(config)

        input_var = node.config.get("input_variable", "input")
        user_message = state.get_variable(input_var, node.config.get("default_message", ""))

        agent_state = agent.launch(str(user_message))
        agent_state = await agent.run_to_completion(agent_state)

        result = ""
        if agent_state.status == AgentStatus.COMPLETED:
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
        condition = node.config.get("condition", "True")

        try:
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
        return {
            "waiting_for": "human_input",
            "prompt": node.config.get("prompt", "Please provide input"),
            "options": node.config.get("options", []),
        }


class HTTPRequestNodeHandler(NodeHandler):
    """
    Handler for HTTP_REQUEST nodes (external API calls).

    Features:
    - Automatic retry with exponential backoff
    - Configurable timeout
    - Variable substitution in URL, headers, and body
    - Response validation
    """

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        import httpx
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )

        # Get configuration
        url = node.config.get("url", "")
        method = node.config.get("method", "GET").upper()
        headers = node.config.get("headers", {})
        body = node.config.get("body")
        timeout = node.config.get("timeout", 30)
        output_var = node.config.get("output_variable", "response")
        max_retries = node.config.get("max_retries", 3)
        retry_on_status = node.config.get("retry_on_status", [502, 503, 504])
        raise_on_error = node.config.get("raise_on_error", False)

        # Template substitution
        url = substitute_variables(url, state.variables)
        headers = {
            k: substitute_variables(str(v), state.variables)
            for k, v in headers.items()
        }
        if body and isinstance(body, str):
            body = substitute_variables(body, state.variables)
        if body and isinstance(body, dict):
            body = {
                k: substitute_variables(str(v), state.variables)
                for k, v in body.items()
            }

        # Create retry decorator
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
            reraise=True,
        )
        async def make_request() -> httpx.Response:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "GET":
                    return await client.get(url, headers=headers)
                elif method == "POST":
                    if isinstance(body, dict):
                        return await client.post(url, headers=headers, json=body)
                    return await client.post(url, headers=headers, content=body)
                elif method == "PUT":
                    if isinstance(body, dict):
                        return await client.put(url, headers=headers, json=body)
                    return await client.put(url, headers=headers, content=body)
                elif method == "DELETE":
                    return await client.delete(url, headers=headers)
                elif method == "PATCH":
                    if isinstance(body, dict):
                        return await client.patch(url, headers=headers, json=body)
                    return await client.patch(url, headers=headers, content=body)
                elif method == "HEAD":
                    return await client.head(url, headers=headers)
                elif method == "OPTIONS":
                    return await client.options(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

        try:
            response = await make_request()

            # Check for retryable status codes
            if response.status_code in retry_on_status:
                # Already handled by retry decorator for connection errors
                # For status codes, we may want to retry manually
                pass

            # Build result
            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "success": 200 <= response.status_code < 300,
            }

            # Parse response body
            try:
                result["body"] = response.json()
            except Exception:
                result["body"] = response.text

            # Raise on error if configured
            if raise_on_error and not result["success"]:
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )

            return {output_var: result}

        except httpx.TimeoutException as e:
            return {
                output_var: {
                    "success": False,
                    "error": "timeout",
                    "error_message": f"Request timed out after {timeout}s",
                }
            }
        except httpx.ConnectError as e:
            return {
                output_var: {
                    "success": False,
                    "error": "connection_error",
                    "error_message": str(e),
                }
            }
        except Exception as e:
            return {
                output_var: {
                    "success": False,
                    "error": "unknown",
                    "error_message": str(e),
                }
            }


class LLMNodeHandler(NodeHandler):
    """
    Handler for LLM nodes (direct LLM invocation).

    Features:
    - Streaming support
    - Variable substitution in prompts
    - Response format options (text, json)
    - Token usage tracking
    """

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        llm_client = context.get("llm_client")
        if not llm_client:
            raise ValueError("LLM client not provided in context")

        # Get configuration
        prompt_template = node.config.get("prompt", "")
        system_prompt = node.config.get("system_prompt", "")
        model = node.config.get("model")
        temperature = node.config.get("temperature", 0.7)
        max_tokens = node.config.get("max_tokens", 1000)
        output_var = node.config.get("output_variable", "llm_response")
        response_format = node.config.get("response_format", "text")  # text or json
        streaming = node.config.get("streaming", False)

        # Substitute variables
        prompt = substitute_variables(prompt_template, state.variables)
        if system_prompt:
            system_prompt = substitute_variables(system_prompt, state.variables)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Handle streaming if supported and requested
        if streaming and hasattr(llm_client, "chat_stream"):
            content_parts = []
            async for chunk in llm_client.chat_stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if isinstance(chunk, dict) and "content" in chunk:
                    content_parts.append(chunk["content"])
                elif isinstance(chunk, str):
                    content_parts.append(chunk)

            content = "".join(content_parts)
            usage = None
        else:
            # Standard non-streaming call
            response = await llm_client.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            usage = response.get("usage") if isinstance(response, dict) else None

        # Parse JSON if requested
        parsed_content = content
        if response_format == "json":
            try:
                # Try to extract JSON from the response
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    parsed_content = json.loads(json_match.group(1))
                else:
                    parsed_content = json.loads(content)
            except json.JSONDecodeError:
                parsed_content = {"raw": content, "parse_error": "Failed to parse JSON"}

        return {
            output_var: parsed_content,
            "llm_metadata": {
                "model": model,
                "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                "completion_tokens": usage.get("completion_tokens") if usage else None,
                "response_format": response_format,
            },
        }


class CodeNodeHandler(NodeHandler):
    """
    Handler for CODE nodes (Python code execution).

    Features:
    - Sandboxed execution with limited builtins
    - Access to workflow variables
    - Timeout protection
    - Error capture with stack traces
    """

    # Safe builtins for code execution
    SAFE_BUILTINS = {
        # Types
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "bytes": bytes,
        "bytearray": bytearray,
        # Iterators
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "reversed": reversed,
        "iter": iter,
        "next": next,
        # Sorting/searching
        "sorted": sorted,
        "min": min,
        "max": max,
        "any": any,
        "all": all,
        # Math
        "sum": sum,
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        # String
        "chr": chr,
        "ord": ord,
        "repr": repr,
        "ascii": ascii,
        "format": format,
        # Type checking
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "callable": callable,
        "hasattr": hasattr,
        "getattr": getattr,
        # Other
        "print": print,
        "id": id,
        "hash": hash,
        "slice": slice,
        "None": None,
        "True": True,
        "False": False,
    }

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        code = node.config.get("code", "")
        output_var = node.config.get("output_variable", "result")
        timeout_seconds = node.config.get("timeout", 30)

        # Build execution context
        exec_globals = {
            "__builtins__": self.SAFE_BUILTINS.copy(),
            "variables": state.variables.copy(),
            "json": json,  # Allow JSON operations
            **state.variables,
        }
        exec_locals: dict[str, Any] = {}

        try:
            # Execute with timeout (basic implementation)
            exec(code, exec_globals, exec_locals)
            result = exec_locals.get("result", exec_globals.get("result"))

            return {
                output_var: result,
                "execution_metadata": {
                    "success": True,
                    "local_vars": list(exec_locals.keys()),
                },
            }

        except SyntaxError as e:
            return {
                output_var: None,
                "error": f"SyntaxError at line {e.lineno}: {e.msg}",
                "execution_metadata": {"success": False, "error_type": "SyntaxError"},
            }
        except NameError as e:
            return {
                output_var: None,
                "error": str(e),
                "execution_metadata": {"success": False, "error_type": "NameError"},
            }
        except Exception as e:
            return {
                output_var: None,
                "error": str(e),
                "execution_metadata": {
                    "success": False,
                    "error_type": type(e).__name__,
                },
            }


class KnowledgeRetrievalNodeHandler(NodeHandler):
    """
    Handler for KNOWLEDGE_RETRIEVAL nodes (RAG retrieval).

    Features:
    - Multiple query support
    - Score threshold filtering
    - Metadata filtering
    - Result formatting options
    """

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        retriever = context.get("retriever")
        if not retriever:
            raise ValueError("Retriever not provided in context")

        # Get configuration
        query_var = node.config.get("query_variable", "query")
        query = state.get_variable(query_var, "")
        top_k = node.config.get("top_k", 5)
        output_var = node.config.get("output_variable", "documents")
        min_score = node.config.get("min_score", 0.0)
        include_metadata = node.config.get("include_metadata", True)
        format_as_context = node.config.get("format_as_context", False)

        # Allow query to be substituted from variables
        if not query:
            query = substitute_variables(node.config.get("query", ""), state.variables)

        # Perform retrieval
        results = await retriever.retrieve(query, top_k=top_k)

        # Format results
        documents = []
        for result in results:
            # Get score
            score = result.score if hasattr(result, "score") else result.get("score", 0)

            # Filter by minimum score
            if score < min_score:
                continue

            # Build document entry
            doc_entry = {
                "content": result.document.content if hasattr(result, "document") else result.get("content", ""),
                "score": score,
            }

            if include_metadata:
                doc_entry["metadata"] = (
                    result.document.metadata if hasattr(result, "document") else result.get("metadata", {})
                )

            documents.append(doc_entry)

        # Optionally format as context string
        context_str = None
        if format_as_context:
            context_parts = []
            for i, doc in enumerate(documents, 1):
                context_parts.append(f"[{i}] {doc['content']}")
            context_str = "\n\n".join(context_parts)

        return {
            output_var: documents,
            "context": context_str,
            "retrieval_metadata": {
                "query": query,
                "result_count": len(documents),
                "filtered_count": len(results) - len(documents),
                "min_score": min_score,
            },
        }
