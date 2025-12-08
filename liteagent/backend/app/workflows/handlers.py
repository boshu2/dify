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


class HTTPRequestNodeHandler(NodeHandler):
    """Handler for HTTP_REQUEST nodes (external API calls)."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        import httpx

        # Get configuration
        url = node.config.get("url", "")
        method = node.config.get("method", "GET").upper()
        headers = node.config.get("headers", {})
        body = node.config.get("body")
        timeout = node.config.get("timeout", 30)
        output_var = node.config.get("output_variable", "response")

        # Template substitution for URL and body
        url = self._substitute_variables(url, state.variables)
        if body and isinstance(body, str):
            body = self._substitute_variables(body, state.variables)
        if body and isinstance(body, dict):
            body = {k: self._substitute_variables(str(v), state.variables) for k, v in body.items()}

        # Make HTTP request
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                response = await client.get(url, headers=headers)
            elif method == "POST":
                if isinstance(body, dict):
                    response = await client.post(url, headers=headers, json=body)
                else:
                    response = await client.post(url, headers=headers, content=body)
            elif method == "PUT":
                if isinstance(body, dict):
                    response = await client.put(url, headers=headers, json=body)
                else:
                    response = await client.put(url, headers=headers, content=body)
            elif method == "DELETE":
                response = await client.delete(url, headers=headers)
            elif method == "PATCH":
                if isinstance(body, dict):
                    response = await client.patch(url, headers=headers, json=body)
                else:
                    response = await client.patch(url, headers=headers, content=body)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

        # Parse response
        result = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

        # Try to parse JSON response
        try:
            result["body"] = response.json()
        except Exception:
            result["body"] = response.text

        return {output_var: result}

    @staticmethod
    def _substitute_variables(template: str, variables: dict[str, Any]) -> str:
        """Substitute {{variable}} placeholders in template."""
        import re
        pattern = r"\{\{(\w+)\}\}"
        def replacer(match):
            var_name = match.group(1)
            return str(variables.get(var_name, match.group(0)))
        return re.sub(pattern, replacer, template)


class LLMNodeHandler(NodeHandler):
    """Handler for LLM nodes (direct LLM invocation)."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # Get LLM client from context
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

        # Substitute variables in prompts
        prompt = self._substitute_variables(prompt_template, state.variables)
        if system_prompt:
            system_prompt = self._substitute_variables(system_prompt, state.variables)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call LLM
        response = await llm_client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract content
        content = response.get("content", "") if isinstance(response, dict) else str(response)

        return {
            output_var: content,
            "llm_metadata": {
                "model": model,
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens") if isinstance(response, dict) else None,
                "completion_tokens": response.get("usage", {}).get("completion_tokens") if isinstance(response, dict) else None,
            },
        }

    @staticmethod
    def _substitute_variables(template: str, variables: dict[str, Any]) -> str:
        """Substitute {{variable}} placeholders in template."""
        import re
        pattern = r"\{\{(\w+)\}\}"
        def replacer(match):
            var_name = match.group(1)
            return str(variables.get(var_name, match.group(0)))
        return re.sub(pattern, replacer, template)


class CodeNodeHandler(NodeHandler):
    """Handler for CODE nodes (Python code execution)."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        code = node.config.get("code", "")
        output_var = node.config.get("output_variable", "result")

        # Create execution context with workflow variables
        # Put variables in globals so lambdas and comprehensions can access them
        exec_globals = {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "print": print,
                "isinstance": isinstance,
                "type": type,
            },
            "variables": state.variables.copy(),
            **state.variables,  # Make variables directly accessible
        }
        exec_locals: dict[str, Any] = {}

        # Execute code
        try:
            exec(code, exec_globals, exec_locals)
            # Result can be in locals or globals
            result = exec_locals.get("result", exec_globals.get("result"))
        except Exception as e:
            return {output_var: None, "error": str(e)}

        return {output_var: result}


class KnowledgeRetrievalNodeHandler(NodeHandler):
    """Handler for KNOWLEDGE_RETRIEVAL nodes (RAG retrieval)."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # Get retriever from context
        retriever = context.get("retriever")
        if not retriever:
            raise ValueError("Retriever not provided in context")

        # Get configuration
        query_var = node.config.get("query_variable", "query")
        query = state.get_variable(query_var, "")
        top_k = node.config.get("top_k", 5)
        output_var = node.config.get("output_variable", "documents")

        # Perform retrieval
        results = await retriever.retrieve(query, top_k=top_k)

        # Format results
        documents = []
        for result in results:
            documents.append({
                "content": result.document.content if hasattr(result, "document") else result.get("content", ""),
                "score": result.score if hasattr(result, "score") else result.get("score", 0),
                "metadata": result.document.metadata if hasattr(result, "document") else result.get("metadata", {}),
            })

        return {
            output_var: documents,
            "retrieval_metadata": {
                "query": query,
                "result_count": len(documents),
            },
        }
