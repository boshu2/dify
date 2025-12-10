"""
Code transformers for different programming languages.
"""

import base64
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class CodeLanguage(str, Enum):
    """Supported programming languages."""

    PYTHON3 = "python3"
    JAVASCRIPT = "javascript"
    JINJA2 = "jinja2"


class CodeTransformer(ABC):
    """Base class for code transformers."""

    @property
    @abstractmethod
    def language(self) -> CodeLanguage:
        """Language this transformer handles."""
        pass

    @abstractmethod
    def transform(
        self,
        code: str,
        inputs: dict[str, Any] | None = None,
        preload: str | None = None,
    ) -> tuple[str, str | None]:
        """Transform user code for safe execution."""
        pass

    def _encode_inputs(self, inputs: dict[str, Any] | None) -> str:
        """Encode inputs as base64 JSON."""
        return base64.b64encode(json.dumps(inputs or {}).encode()).decode()


class PythonTransformer(CodeTransformer):
    """Transformer for Python 3 code."""

    TEMPLATE = '''
import json
import base64
import sys
from io import StringIO

# Capture output
_output_buffer = StringIO()
_original_stdout = sys.stdout

def _safe_print(*args, **kwargs):
    """Safe print that captures to buffer."""
    print(*args, file=_output_buffer, **kwargs)

# Override print
print = _safe_print

# Decode inputs
try:
    _inputs_json = base64.b64decode("{inputs_b64}").decode()
    inputs = json.loads(_inputs_json)
except Exception as e:
    inputs = {{}}

{preload}

# User code
{code}

# Execute main and capture result
_result = None
_error = None

try:
    if "main" in dir():
        if inputs:
            _result = main(**inputs)
        else:
            _result = main()
except Exception as e:
    _error = str(e)

# Restore stdout
sys.stdout = _original_stdout

# Output result
if _error:
    print(f"<<ERROR>>{{_error}}<<ERROR>>")
else:
    try:
        _result_json = json.dumps(_result, default=str)
        print(f"<<RESULT>>{{_result_json}}<<RESULT>>")
    except Exception as e:
        print(f"<<ERROR>>Failed to serialize result: {{e}}<<ERROR>>")

# Output captured stdout
_captured = _output_buffer.getvalue()
if _captured:
    print(f"<<STDOUT>>{{_captured}}<<STDOUT>>")
'''

    @property
    def language(self) -> CodeLanguage:
        return CodeLanguage.PYTHON3

    def transform(
        self,
        code: str,
        inputs: dict[str, Any] | None = None,
        preload: str | None = None,
    ) -> tuple[str, str | None]:
        """Transform Python code."""
        inputs_b64 = self._encode_inputs(inputs)

        transformed = self.TEMPLATE.format(
            inputs_b64=inputs_b64,
            preload=preload or "# No preload",
            code=code,
        )

        return transformed, None


class JavaScriptTransformer(CodeTransformer):
    """Transformer for JavaScript/Node.js code."""

    TEMPLATE = '''
// Input decoding
let inputs = {{}};
try {{
    const inputsJson = Buffer.from("{inputs_b64}", "base64").toString("utf8");
    inputs = JSON.parse(inputsJson);
}} catch (e) {{
    inputs = {{}};
}}

// Capture console.log
const _logs = [];
const _originalLog = console.log;
console.log = (...args) => {{
    _logs.push(args.map(a => typeof a === "object" ? JSON.stringify(a) : String(a)).join(" "));
}};

{preload}

// User code
{code}

// Execute main and capture result
let _result = null;
let _error = null;

try {{
    if (typeof main === "function") {{
        _result = main(inputs);
    }}
}} catch (e) {{
    _error = e.message || String(e);
}}

// Restore console
console.log = _originalLog;

// Output result
if (_error) {{
    console.log("<<ERROR>>" + _error + "<<ERROR>>");
}} else {{
    try {{
        const resultJson = JSON.stringify(_result);
        console.log("<<RESULT>>" + resultJson + "<<RESULT>>");
    }} catch (e) {{
        console.log("<<ERROR>>Failed to serialize result: " + e.message + "<<ERROR>>");
    }}
}}

// Output captured logs
if (_logs.length > 0) {{
    console.log("<<STDOUT>>" + _logs.join("\\n") + "<<STDOUT>>");
}}
'''

    @property
    def language(self) -> CodeLanguage:
        return CodeLanguage.JAVASCRIPT

    def transform(
        self,
        code: str,
        inputs: dict[str, Any] | None = None,
        preload: str | None = None,
    ) -> tuple[str, str | None]:
        """Transform JavaScript code."""
        inputs_b64 = self._encode_inputs(inputs)

        transformed = self.TEMPLATE.format(
            inputs_b64=inputs_b64,
            preload=preload or "// No preload",
            code=code,
        )

        return transformed, None


class Jinja2Transformer(CodeTransformer):
    """Transformer for Jinja2 templates."""

    @property
    def language(self) -> CodeLanguage:
        return CodeLanguage.JINJA2

    def transform(
        self,
        code: str,
        inputs: dict[str, Any] | None = None,
        preload: str | None = None,
    ) -> tuple[str, str | None]:
        """Transform Jinja2 template - returns template and inputs separately."""
        # Jinja2 templates are executed directly, no transformation needed
        return code, None


# Transformer registry
_transformers: dict[CodeLanguage, CodeTransformer] = {}


def register_transformer(transformer: CodeTransformer) -> None:
    """Register a code transformer."""
    _transformers[transformer.language] = transformer


def get_transformer(language: CodeLanguage | str) -> CodeTransformer:
    """Get transformer for a language."""
    if isinstance(language, str):
        language = CodeLanguage(language)

    if language not in _transformers:
        raise ValueError(f"No transformer for language: {language}")

    return _transformers[language]


# Register default transformers
register_transformer(PythonTransformer())
register_transformer(JavaScriptTransformer())
register_transformer(Jinja2Transformer())
