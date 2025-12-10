"""
Code executor with multiple execution backends.

Supports:
- Local execution with RestrictedPython
- Docker container execution
- Remote sandbox API
"""

import asyncio
import base64
import json
import re
import signal
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from app.core.sandbox.config import SandboxConfig, get_sandbox_config


class CodeLanguage(str, Enum):
    """Supported programming languages."""

    PYTHON3 = "python3"
    JAVASCRIPT = "javascript"
    JINJA2 = "jinja2"


class ExecutionError(Exception):
    """Error during code execution."""

    def __init__(
        self,
        message: str,
        error_type: str = "ExecutionError",
        line_number: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.line_number = line_number
        self.details = details or {}


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    error_type: str | None = None
    line_number: int | None = None
    execution_time_ms: float = 0
    memory_used_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "error_type": self.error_type,
            "line_number": self.line_number,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_bytes": self.memory_used_bytes,
            "metadata": self.metadata,
        }


class CodeExecutor:
    """
    Safe code executor with multiple backends.

    Supports:
    - Local: Uses RestrictedPython for safe execution
    - Docker: Runs code in isolated container
    - Remote: Calls external sandbox API
    """

    # Result markers for extracting output
    RESULT_START = "<<RESULT>>"
    RESULT_END = "<<RESULT>>"

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or get_sandbox_config()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._http_client: httpx.AsyncClient | None = None

    async def execute(
        self,
        language: CodeLanguage | str,
        code: str,
        inputs: dict[str, Any] | None = None,
        preload: str | None = None,
        enable_network: bool | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Execute code safely.

        Args:
            language: Programming language
            code: Code to execute
            inputs: Input variables for the code
            preload: Preload script (e.g., imports)
            enable_network: Override network access setting
            timeout: Execution timeout override

        Returns:
            ExecutionResult with output or error
        """
        if isinstance(language, str):
            language = CodeLanguage(language)

        timeout = timeout or self.config.execution_timeout
        enable_network = enable_network if enable_network is not None else self.config.enable_network

        start_time = datetime.now(timezone.utc)

        try:
            if self.config.mode == "remote":
                result = await self._execute_remote(
                    language, code, inputs, preload, enable_network, timeout
                )
            elif self.config.mode == "docker":
                result = await self._execute_docker(
                    language, code, inputs, preload, enable_network, timeout
                )
            else:
                result = await self._execute_local(
                    language, code, inputs, preload, timeout
                )

            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

            return result

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                error="Execution timed out",
                error_type="TimeoutError",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _execute_local(
        self,
        language: CodeLanguage,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
        timeout: int,
    ) -> ExecutionResult:
        """Execute code locally with restrictions."""
        # Validate code against blocked patterns
        self._validate_code(code)

        if language == CodeLanguage.PYTHON3:
            return await self._execute_python_local(code, inputs, preload, timeout)
        elif language == CodeLanguage.JAVASCRIPT:
            return await self._execute_javascript_local(code, inputs, preload, timeout)
        elif language == CodeLanguage.JINJA2:
            return await self._execute_jinja2_local(code, inputs, timeout)
        else:
            raise ExecutionError(f"Unsupported language: {language}")

    async def _execute_python_local(
        self,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
        timeout: int,
    ) -> ExecutionResult:
        """Execute Python code with RestrictedPython."""
        try:
            from RestrictedPython import compile_restricted, safe_builtins
            from RestrictedPython.Eval import default_guarded_getitem
            from RestrictedPython.Guards import (
                full_write_guard,
                guarded_iter_unpack_sequence,
                guarded_unpack_sequence,
            )
        except ImportError:
            # Fallback to subprocess execution
            return await self._execute_python_subprocess(code, inputs, preload, timeout)

        # Prepare execution environment
        restricted_globals = {
            "__builtins__": safe_builtins,
            "_getitem_": default_guarded_getitem,
            "_write_": full_write_guard,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_unpack_sequence_": guarded_unpack_sequence,
            "_getiter_": iter,
            "_print_": self._safe_print,
        }

        # Add allowed modules
        for module_name in self.config.allowed_modules:
            try:
                module = __import__(module_name)
                restricted_globals[module_name] = module
            except ImportError:
                pass

        # Add inputs
        if inputs:
            restricted_globals["inputs"] = inputs

        # Capture stdout
        stdout_capture: list[str] = []
        restricted_globals["_stdout_"] = stdout_capture

        # Prepare code with main function wrapper
        full_code = self._wrap_python_code(code, preload)

        try:
            # Compile with restrictions
            byte_code = compile_restricted(
                full_code,
                filename="<sandbox>",
                mode="exec",
            )

            if byte_code.errors:
                return ExecutionResult(
                    success=False,
                    error="; ".join(byte_code.errors),
                    error_type="CompilationError",
                )

            # Execute with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._run_restricted_code,
                    byte_code.code,
                    restricted_globals,
                ),
                timeout=timeout,
            )

            return ExecutionResult(
                success=True,
                output=result,
                stdout="\n".join(stdout_capture),
            )

        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type="SyntaxError",
                line_number=e.lineno,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                stderr=traceback.format_exc(),
            )

    def _run_restricted_code(
        self,
        byte_code: Any,
        globals_dict: dict[str, Any],
    ) -> Any:
        """Run compiled restricted code."""
        exec(byte_code, globals_dict)

        # Get result from main function if defined
        if "main" in globals_dict:
            inputs = globals_dict.get("inputs", {})
            return globals_dict["main"](**inputs) if inputs else globals_dict["main"]()
        elif "_result_" in globals_dict:
            return globals_dict["_result_"]

        return None

    async def _execute_python_subprocess(
        self,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
        timeout: int,
    ) -> ExecutionResult:
        """Execute Python code in subprocess (fallback)."""
        # Create wrapped code
        wrapped_code = self._wrap_python_code_for_subprocess(code, inputs, preload)

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(wrapped_code)
            temp_path = f.name

        try:
            # Run in subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                raise

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Extract result from stdout
            output = self._extract_result(stdout_str)

            if process.returncode != 0:
                return ExecutionResult(
                    success=False,
                    error=stderr_str or "Execution failed",
                    error_type="RuntimeError",
                    stdout=stdout_str,
                    stderr=stderr_str,
                )

            return ExecutionResult(
                success=True,
                output=output,
                stdout=stdout_str,
                stderr=stderr_str,
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def _execute_javascript_local(
        self,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
        timeout: int,
    ) -> ExecutionResult:
        """Execute JavaScript code using Node.js subprocess."""
        # Create wrapped code
        wrapped_code = self._wrap_javascript_code(code, inputs, preload)

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".js",
            delete=False,
        ) as f:
            f.write(wrapped_code)
            temp_path = f.name

        try:
            # Run with node
            process = await asyncio.create_subprocess_exec(
                "node",
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                raise

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Extract result
            output = self._extract_result(stdout_str)

            if process.returncode != 0:
                return ExecutionResult(
                    success=False,
                    error=stderr_str or "Execution failed",
                    error_type="RuntimeError",
                    stdout=stdout_str,
                    stderr=stderr_str,
                )

            return ExecutionResult(
                success=True,
                output=output,
                stdout=stdout_str,
                stderr=stderr_str,
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def _execute_jinja2_local(
        self,
        template: str,
        inputs: dict[str, Any] | None,
        timeout: int,
    ) -> ExecutionResult:
        """Execute Jinja2 template."""
        try:
            from jinja2 import Environment, StrictUndefined
            from jinja2.sandbox import SandboxedEnvironment

            env = SandboxedEnvironment(undefined=StrictUndefined)
            compiled = env.from_string(template)

            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    compiled.render,
                    inputs or {},
                ),
                timeout=timeout,
            )

            return ExecutionResult(
                success=True,
                output=result,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _execute_remote(
        self,
        language: CodeLanguage,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
        enable_network: bool,
        timeout: int,
    ) -> ExecutionResult:
        """Execute code via remote sandbox API."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.config.remote_endpoint,
                timeout=httpx.Timeout(
                    connect=self.config.connection_timeout,
                    read=timeout + 5,  # Give sandbox extra time
                    write=10.0,
                    pool=5.0,
                ),
            )

        # Transform code for remote execution
        from app.core.sandbox.transformers import get_transformer

        transformer = get_transformer(language)
        transformed_code, transformed_preload = transformer.transform(
            code, inputs, preload
        )

        try:
            response = await self._http_client.post(
                "/v1/sandbox/run",
                json={
                    "language": language.value,
                    "code": transformed_code,
                    "preload": transformed_preload or "",
                    "enable_network": enable_network,
                },
                headers={"X-Api-Key": self.config.remote_api_key},
            )

            if response.status_code == 503:
                raise ExecutionError("Sandbox service unavailable", "ServiceError")

            data = response.json()

            if data.get("code") != 0:
                return ExecutionResult(
                    success=False,
                    error=data.get("message", "Unknown error"),
                    error_type="RemoteExecutionError",
                )

            result_data = data.get("data", {})
            stdout = result_data.get("stdout", "")
            stderr = result_data.get("error", "")

            # Extract result from stdout
            output = self._extract_result(stdout)

            return ExecutionResult(
                success=True,
                output=output,
                stdout=stdout,
                stderr=stderr,
            )

        except httpx.RequestError as e:
            raise ExecutionError(f"Connection error: {e}", "ConnectionError")

    async def _execute_docker(
        self,
        language: CodeLanguage,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
        enable_network: bool,
        timeout: int,
    ) -> ExecutionResult:
        """Execute code in Docker container."""
        # Transform code
        from app.core.sandbox.transformers import get_transformer

        transformer = get_transformer(language)
        transformed_code, _ = transformer.transform(code, inputs, preload)

        # Determine file extension
        ext = ".py" if language == CodeLanguage.PYTHON3 else ".js"
        cmd = ["python3"] if language == CodeLanguage.PYTHON3 else ["node"]

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=ext,
            delete=False,
        ) as f:
            f.write(transformed_code)
            temp_path = f.name

        try:
            # Build docker command
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--network", "none" if not enable_network else self.config.docker_network,
                "--memory", f"{self.config.max_memory_mb}m",
                "--cpus", "1",
                "--user", "nobody",
                "--read-only",
                "-v", f"{temp_path}:/code/script{ext}:ro",
                "-w", "/code",
                self.config.docker_image,
                *cmd,
                f"script{ext}",
            ]

            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout + 5,  # Extra time for Docker overhead
                )
            except asyncio.TimeoutError:
                # Kill container
                subprocess.run(
                    ["docker", "kill", "-s", "SIGKILL"],
                    capture_output=True,
                )
                raise

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            output = self._extract_result(stdout_str)

            if process.returncode != 0:
                return ExecutionResult(
                    success=False,
                    error=stderr_str or "Container execution failed",
                    error_type="DockerExecutionError",
                    stdout=stdout_str,
                    stderr=stderr_str,
                )

            return ExecutionResult(
                success=True,
                output=output,
                stdout=stdout_str,
                stderr=stderr_str,
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _validate_code(self, code: str) -> None:
        """Validate code against blocked patterns."""
        for pattern in self.config.blocked_patterns:
            if pattern in code:
                raise ExecutionError(
                    f"Blocked pattern detected: {pattern}",
                    "SecurityError",
                )

    def _wrap_python_code(self, code: str, preload: str | None) -> str:
        """Wrap Python code for safe execution."""
        parts = []

        if preload:
            parts.append(preload)

        parts.append(code)

        return "\n".join(parts)

    def _wrap_python_code_for_subprocess(
        self,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
    ) -> str:
        """Wrap Python code for subprocess execution."""
        inputs_b64 = base64.b64encode(
            json.dumps(inputs or {}).encode()
        ).decode()

        template = f'''
import json
import base64

# Decode inputs
inputs_json = base64.b64decode("{inputs_b64}").decode()
inputs = json.loads(inputs_json)

{preload or "# No preload"}

{code}

# Execute main function if defined
if "main" in dir():
    result = main(**inputs) if inputs else main()
    result_json = json.dumps(result, default=str)
    print(f"<<RESULT>>{{result_json}}<<RESULT>>")
'''
        return template

    def _wrap_javascript_code(
        self,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
    ) -> str:
        """Wrap JavaScript code for execution."""
        inputs_b64 = base64.b64encode(
            json.dumps(inputs or {}).encode()
        ).decode()

        template = f'''
// Decode inputs
const inputs = JSON.parse(Buffer.from("{inputs_b64}", "base64").toString());

{preload or "// No preload"}

{code}

// Execute main function if defined
if (typeof main === "function") {{
    const result = main(inputs);
    const resultJson = JSON.stringify(result);
    console.log("<<RESULT>>" + resultJson + "<<RESULT>>");
}}
'''
        return template

    def _extract_result(self, output: str) -> Any:
        """Extract result from output markers."""
        pattern = f"{self.RESULT_START}(.+?){self.RESULT_END}"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return match.group(1)

        return None

    def _safe_print(self, *args: Any, **kwargs: Any) -> None:
        """Safe print function for RestrictedPython."""
        pass  # Captured by stdout

    async def close(self) -> None:
        """Close executor and release resources."""
        if self._http_client:
            await self._http_client.aclose()
        self._executor.shutdown(wait=False)


# Global executor instance
_executor: CodeExecutor | None = None


def get_executor() -> CodeExecutor:
    """Get global code executor."""
    global _executor
    if _executor is None:
        _executor = CodeExecutor()
    return _executor
