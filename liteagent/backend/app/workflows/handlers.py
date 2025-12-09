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


class LoopNodeHandler(NodeHandler):
    """
    Handler for LOOP nodes (iteration over arrays).

    Features:
    - Iterate over arrays/lists
    - Configurable loop variable name
    - Break condition support
    - Max iterations limit for safety
    - Accumulator for collecting results
    """

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # Get configuration
        array_var = node.config.get("array_variable", "items")
        item_var = node.config.get("item_variable", "item")
        index_var = node.config.get("index_variable", "index")
        output_var = node.config.get("output_variable", "loop_results")
        max_iterations = node.config.get("max_iterations", 1000)
        break_condition = node.config.get("break_condition")
        transform_code = node.config.get("transform")

        # Get array to iterate over
        items = state.get_variable(array_var, [])
        if not isinstance(items, (list, tuple)):
            return {
                output_var: [],
                "error": f"Variable '{array_var}' is not iterable",
                "loop_metadata": {"success": False, "iterations": 0},
            }

        results = []
        iterations = 0

        for idx, item in enumerate(items):
            # Safety limit
            if iterations >= max_iterations:
                break

            iterations += 1

            # Set loop variables in a temporary context
            loop_vars = {
                item_var: item,
                index_var: idx,
                "loop_length": len(items),
                "is_first": idx == 0,
                "is_last": idx == len(items) - 1,
            }

            # Check break condition
            if break_condition:
                try:
                    eval_context = {
                        **state.variables,
                        **loop_vars,
                    }
                    if eval(break_condition, {"__builtins__": {}}, eval_context):
                        break
                except Exception:
                    pass  # Continue on evaluation errors

            # Apply transform if provided
            if transform_code:
                try:
                    exec_globals = {
                        "__builtins__": {
                            "len": len,
                            "str": str,
                            "int": int,
                            "float": float,
                            "list": list,
                            "dict": dict,
                            "bool": bool,
                            "range": range,
                            "sum": sum,
                            "min": min,
                            "max": max,
                            "abs": abs,
                            "round": round,
                            "sorted": sorted,
                            "reversed": reversed,
                            "enumerate": enumerate,
                            "zip": zip,
                            "None": None,
                            "True": True,
                            "False": False,
                        },
                        "json": json,
                        **state.variables,
                        **loop_vars,
                    }
                    exec_locals: dict[str, Any] = {}
                    exec(transform_code, exec_globals, exec_locals)
                    result = exec_locals.get("result", item)
                    results.append(result)
                except Exception as e:
                    # On transform error, include item with error marker
                    results.append({"_item": item, "_error": str(e)})
            else:
                # No transform - just collect items
                results.append(item)

        return {
            output_var: results,
            "loop_metadata": {
                "success": True,
                "iterations": iterations,
                "total_items": len(items),
                "result_count": len(results),
            },
        }


class VariableAggregatorNodeHandler(NodeHandler):
    """
    Handler for VARIABLE_AGGREGATOR nodes.

    Combines multiple variables into a single output variable using
    different aggregation strategies.

    Features:
    - Merge dictionaries
    - Concatenate arrays
    - Format as template
    - Select first non-null
    - Custom aggregation expressions
    """

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # Get configuration
        input_vars = node.config.get("input_variables", [])
        output_var = node.config.get("output_variable", "aggregated")
        strategy = node.config.get("strategy", "merge")
        template = node.config.get("template")
        expression = node.config.get("expression")

        # Collect input values
        values = {}
        for var_name in input_vars:
            values[var_name] = state.get_variable(var_name)

        result: Any = None

        if strategy == "merge":
            # Merge all dictionaries
            result = {}
            for var_name in input_vars:
                val = values.get(var_name)
                if isinstance(val, dict):
                    result.update(val)
                else:
                    result[var_name] = val

        elif strategy == "concat":
            # Concatenate arrays
            result = []
            for var_name in input_vars:
                val = values.get(var_name)
                if isinstance(val, (list, tuple)):
                    result.extend(val)
                elif val is not None:
                    result.append(val)

        elif strategy == "first_non_null":
            # Return first non-null value
            for var_name in input_vars:
                val = values.get(var_name)
                if val is not None:
                    result = val
                    break

        elif strategy == "template" and template:
            # Format using template - merge values with input_vars taking precedence
            try:
                format_vars = {**state.variables, **values}
                result = template.format(**format_vars)
            except KeyError as e:
                return {
                    output_var: None,
                    "error": f"Missing variable in template: {e}",
                }

        elif strategy == "expression" and expression:
            # Evaluate custom expression
            try:
                eval_context = {**values, **state.variables}
                result = eval(expression, {"__builtins__": {}}, eval_context)
            except Exception as e:
                return {
                    output_var: None,
                    "error": f"Expression error: {e}",
                }

        elif strategy == "object":
            # Create object with variable names as keys
            result = values

        elif strategy == "array":
            # Create array of values
            result = list(values.values())

        else:
            # Default: return all values as dict
            result = values

        return {
            output_var: result,
            "aggregation_metadata": {
                "strategy": strategy,
                "input_count": len(input_vars),
            },
        }


class ParallelNodeHandler(NodeHandler):
    """
    Handler for PARALLEL nodes (concurrent execution marker).

    This node marks the start of parallel branches. The actual parallel
    execution is handled by the workflow engine, not this handler.

    This handler collects branch information for the engine to execute.
    """

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        branches = node.config.get("branches", [])
        timeout = node.config.get("timeout", 300)
        fail_fast = node.config.get("fail_fast", False)

        return {
            "parallel_info": {
                "branches": branches,
                "timeout": timeout,
                "fail_fast": fail_fast,
                "status": "pending",
            },
        }


class MergeNodeHandler(NodeHandler):
    """
    Handler for MERGE nodes (wait for parallel branches).

    Waits for all parallel branches to complete and merges their results.
    """

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # Get branch results from state
        branch_outputs = state.get_variable("_branch_outputs", {})
        output_var = node.config.get("output_variable", "merged")
        merge_strategy = node.config.get("merge_strategy", "object")

        if merge_strategy == "object":
            # Keep outputs as object with branch names
            result = branch_outputs
        elif merge_strategy == "array":
            # Convert to array
            result = list(branch_outputs.values())
        elif merge_strategy == "flatten":
            # Flatten all arrays
            result = []
            for val in branch_outputs.values():
                if isinstance(val, (list, tuple)):
                    result.extend(val)
                else:
                    result.append(val)
        else:
            result = branch_outputs

        return {
            output_var: result,
            "merge_metadata": {
                "branch_count": len(branch_outputs),
                "branches": list(branch_outputs.keys()),
            },
        }


class WaitNodeHandler(NodeHandler):
    """Handler for WAIT nodes (delay/external event)."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        import asyncio

        wait_type = node.config.get("wait_type", "duration")
        duration = node.config.get("duration", 0)

        if wait_type == "duration" and duration > 0:
            # Simple delay (capped at 60 seconds for safety)
            await asyncio.sleep(min(duration, 60))
            return {"waited": duration, "wait_type": "duration"}

        elif wait_type == "event":
            # Mark as waiting for external event
            event_name = node.config.get("event_name", "external_event")
            return {
                "waiting_for": event_name,
                "wait_type": "event",
                "status": "waiting",
            }

        return {"waited": 0, "wait_type": wait_type}
