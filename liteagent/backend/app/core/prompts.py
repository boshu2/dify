"""
Prompt template system for LLM agents.
Provides versioned, parameterized prompts with validation.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from string import Template
from typing import Any


class TemplateFormat(str, Enum):
    """Supported template formats."""

    PYTHON = "python"  # {variable} format
    JINJA2 = "jinja2"  # {{ variable }} format
    FSTRING = "fstring"  # Standard Python f-string style


@dataclass
class PromptVariable:
    """Definition of a prompt variable."""

    name: str
    description: str = ""
    required: bool = True
    default: Any = None
    var_type: type = str
    enum: list[Any] | None = None

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """
        Validate a value for this variable.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if value is None:
            if self.required and self.default is None:
                return False, f"Required variable '{self.name}' is missing"
            return True, None

        if self.enum and value not in self.enum:
            return False, f"Value '{value}' not in allowed values: {self.enum}"

        if not isinstance(value, self.var_type):
            try:
                self.var_type(value)
            except (ValueError, TypeError):
                return False, f"Cannot convert '{value}' to {self.var_type.__name__}"

        return True, None


@dataclass
class PromptTemplate:
    """A parameterized prompt template."""

    name: str
    template: str
    description: str = ""
    variables: list[PromptVariable] = field(default_factory=list)
    format: TemplateFormat = TemplateFormat.PYTHON
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def _extract_variables(self) -> set[str]:
        """Extract variable names from template."""
        if self.format == TemplateFormat.PYTHON:
            # Match {variable} but not {{escaped}}
            pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
            return set(re.findall(pattern, self.template))

        elif self.format == TemplateFormat.JINJA2:
            # Match {{ variable }}
            pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
            return set(re.findall(pattern, self.template))

        elif self.format == TemplateFormat.FSTRING:
            # Same as Python format
            pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
            return set(re.findall(pattern, self.template))

        return set()

    def get_variable_names(self) -> list[str]:
        """Get names of all variables in template."""
        defined = {v.name for v in self.variables}
        extracted = self._extract_variables()
        return sorted(defined | extracted)

    def validate_variables(self, values: dict[str, Any]) -> list[str]:
        """
        Validate provided variable values.

        Returns:
            List of validation error messages.
        """
        errors = []

        # Check defined variables
        for var in self.variables:
            value = values.get(var.name, var.default)
            is_valid, error = var.validate(value)
            if not is_valid:
                errors.append(error)

        # Check for unknown variables
        known_vars = {v.name for v in self.variables}
        for name in values:
            if name not in known_vars and name not in self._extract_variables():
                # Not an error, just a warning - allow extra variables
                pass

        return errors

    def render(self, **kwargs) -> str:
        """
        Render the template with provided values.

        Args:
            **kwargs: Variable values.

        Returns:
            Rendered prompt string.

        Raises:
            ValueError: If required variables are missing.
        """
        # Fill in defaults
        values = {}
        for var in self.variables:
            if var.name in kwargs:
                values[var.name] = kwargs[var.name]
            elif var.default is not None:
                values[var.name] = var.default
            elif not var.required:
                values[var.name] = ""

        # Add any extra kwargs
        values.update(kwargs)

        # Validate
        errors = self.validate_variables(values)
        if errors:
            raise ValueError(f"Template validation failed: {'; '.join(errors)}")

        # Render based on format
        if self.format == TemplateFormat.PYTHON:
            return self.template.format(**values)

        elif self.format == TemplateFormat.JINJA2:
            # Simple Jinja2-like replacement
            result = self.template
            for name, value in values.items():
                result = re.sub(
                    rf"\{{\{{\s*{name}\s*\}}\}}",
                    str(value),
                    result,
                )
            return result

        elif self.format == TemplateFormat.FSTRING:
            return self.template.format(**values)

        return self.template

    def __call__(self, **kwargs) -> str:
        """Allow calling template directly."""
        return self.render(**kwargs)


class PromptBuilder:
    """Builder for constructing prompts from multiple parts."""

    def __init__(self):
        self._parts: list[str] = []
        self._variables: dict[str, Any] = {}

    def add(self, text: str) -> "PromptBuilder":
        """Add text to the prompt."""
        self._parts.append(text)
        return self

    def add_line(self, text: str) -> "PromptBuilder":
        """Add text followed by newline."""
        self._parts.append(text + "\n")
        return self

    def add_section(self, title: str, content: str) -> "PromptBuilder":
        """Add a titled section."""
        self._parts.append(f"\n## {title}\n{content}\n")
        return self

    def add_if(self, condition: bool, text: str) -> "PromptBuilder":
        """Add text only if condition is true."""
        if condition:
            self._parts.append(text)
        return self

    def set_variable(self, name: str, value: Any) -> "PromptBuilder":
        """Set a variable for template rendering."""
        self._variables[name] = value
        return self

    def build(self) -> str:
        """Build the final prompt."""
        prompt = "".join(self._parts)

        if self._variables:
            prompt = prompt.format(**self._variables)

        return prompt

    def clear(self) -> "PromptBuilder":
        """Clear the builder."""
        self._parts.clear()
        self._variables.clear()
        return self


class PromptRegistry:
    """Registry for managing prompt templates."""

    def __init__(self):
        self._templates: dict[str, dict[str, PromptTemplate]] = {}  # name -> version -> template

    def register(self, template: PromptTemplate) -> None:
        """Register a prompt template."""
        if template.name not in self._templates:
            self._templates[template.name] = {}
        self._templates[template.name][template.version] = template

    def get(
        self,
        name: str,
        version: str | None = None,
    ) -> PromptTemplate | None:
        """
        Get a template by name and optional version.

        Args:
            name: Template name.
            version: Specific version (latest if None).

        Returns:
            Template or None if not found.
        """
        if name not in self._templates:
            return None

        versions = self._templates[name]

        if version:
            return versions.get(version)

        # Return latest version
        if versions:
            latest = max(versions.keys())
            return versions[latest]

        return None

    def get_all_versions(self, name: str) -> list[PromptTemplate]:
        """Get all versions of a template."""
        if name not in self._templates:
            return []
        return list(self._templates[name].values())

    def list_templates(self) -> list[str]:
        """List all template names."""
        return list(self._templates.keys())

    def unregister(self, name: str, version: str | None = None) -> None:
        """Unregister a template."""
        if name not in self._templates:
            return

        if version:
            self._templates[name].pop(version, None)
            if not self._templates[name]:
                del self._templates[name]
        else:
            del self._templates[name]

    def __len__(self) -> int:
        """Return total number of templates (all versions)."""
        return sum(len(versions) for versions in self._templates.values())

    def __contains__(self, name: str) -> bool:
        """Check if a template name exists."""
        return name in self._templates


# Pre-built prompt templates
SYSTEM_PROMPTS = {
    "helpful_assistant": PromptTemplate(
        name="helpful_assistant",
        template="You are a helpful AI assistant. Your goal is to assist users with their questions and tasks in a clear, accurate, and friendly manner.",
        description="Basic helpful assistant system prompt",
        version="1.0.0",
    ),
    "code_assistant": PromptTemplate(
        name="code_assistant",
        template="""You are an expert programming assistant. You help users with:
- Writing clean, efficient code
- Debugging issues
- Explaining concepts
- Code reviews and best practices

When writing code:
- Use proper formatting and comments
- Follow language-specific conventions
- Consider edge cases and error handling

Current context: {context}""",
        description="System prompt for coding assistance",
        variables=[
            PromptVariable(
                name="context",
                description="Additional context about the coding task",
                required=False,
                default="General programming assistance",
            ),
        ],
        version="1.0.0",
    ),
    "qa_assistant": PromptTemplate(
        name="qa_assistant",
        template="""You are a question-answering assistant. Answer questions based on the provided context.

Context:
{context}

Guidelines:
- Only answer based on the provided context
- If the answer is not in the context, say so
- Be concise but complete
- Cite relevant parts of the context when applicable""",
        description="Q&A assistant with RAG context",
        variables=[
            PromptVariable(
                name="context",
                description="Retrieved context for answering questions",
                required=True,
            ),
        ],
        version="1.0.0",
    ),
}


# Factory functions
def create_prompt_template(
    name: str,
    template: str,
    description: str = "",
    variables: list[dict[str, Any]] | None = None,
    version: str = "1.0.0",
) -> PromptTemplate:
    """
    Create a prompt template.

    Args:
        name: Template name.
        template: Template string.
        description: Template description.
        variables: Variable definitions.
        version: Template version.

    Returns:
        PromptTemplate instance.
    """
    var_list = []
    if variables:
        for var_def in variables:
            var_list.append(PromptVariable(**var_def))

    return PromptTemplate(
        name=name,
        template=template,
        description=description,
        variables=var_list,
        version=version,
    )


def get_system_prompt(name: str) -> PromptTemplate | None:
    """Get a pre-built system prompt."""
    return SYSTEM_PROMPTS.get(name)
