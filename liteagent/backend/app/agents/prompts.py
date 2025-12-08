"""
Agent prompt templates.

Factor 2: Own your prompts - all prompts in one place, fully controllable.
"""
from app.agents.types import ToolDefinition


class AgentPrompts:
    """
    Factor 2: Own your prompts.
    All prompts in one place, fully controllable.
    """

    @staticmethod
    def system_prompt(agent_purpose: str, tools: list[ToolDefinition]) -> str:
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )
        return f"""You are an AI agent with a specific purpose: {agent_purpose}

Available tools:
{tool_descriptions}

Instructions:
1. Analyze the user's request carefully
2. Break down complex tasks into steps
3. Use tools when needed to accomplish tasks
4. If you need human approval or input, use the request_human_input tool
5. When the task is complete, provide a final response without tool calls

Always think step-by-step before acting."""

    @staticmethod
    def error_context(error: str, tool_name: str) -> str:
        """
        Factor 9: Compact errors into context.
        Keep error messages concise but informative.
        """
        return f"Tool '{tool_name}' failed: {error[:200]}"  # Truncate long errors
