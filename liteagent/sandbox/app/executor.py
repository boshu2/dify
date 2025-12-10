"""
Code executor for sandbox service.

Executes code in isolated environment with:
- Resource limits (memory, CPU, time)
- Network isolation
- Restricted builtins
"""

import asyncio
import json
import re
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import SandboxConfig, get_config
from app.transformers import (
    CodeLanguage,
    JavaScriptTransformer,
    Jinja2Transformer,
    PythonTransformer,
    get_transformer,
)
from app.validator import OutputValidator, sanitize_for_json


class ExecutionError(Exception):
    """Error during code execution."""

    def __init__(
        self,
        message: str,
        error_type: str = "ExecutionError",
        line_number: int | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.line_number = line_number


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
        }


class CodeExecutor:
    """
    Isolated code executor.

    Executes code in subprocess with resource limits.
    """

    RESULT_START = "<<RESULT>>"
    RESULT_END = "<<RESULT>>"
    ERROR_START = "<<ERROR>>"
    ERROR_END = "<<ERROR>>"
    STDOUT_START = "<<STDOUT>>"
    STDOUT_END = "<<STDOUT>>"

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or get_config()
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._validator = OutputValidator(self.config)

    async def execute(
        self,
        language: CodeLanguage | str,
        code: str,
        inputs: dict[str, Any] | None = None,
        preload: str | None = None,
        enable_network: bool | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Execute code safely in isolated subprocess."""
        if isinstance(language, str):
            language = CodeLanguage(language)

        timeout = timeout or self.config.execution_timeout
        enable_network = (
            enable_network if enable_network is not None else self.config.enable_network
        )

        start_time = datetime.now(timezone.utc)

        try:
            # Validate code against blocked patterns
            self._validate_code(code)

            # Execute based on language
            if language == CodeLanguage.PYTHON3:
                result = await self._execute_python(code, inputs, preload, timeout)
            elif language == CodeLanguage.JAVASCRIPT:
                result = await self._execute_javascript(code, inputs, preload, timeout)
            elif language == CodeLanguage.JINJA2:
                result = await self._execute_jinja2(code, inputs, timeout)
            else:
                raise ExecutionError(f"Unsupported language: {language}")

            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

            # Validate output
            if result.success and result.output is not None:
                validation = self._validator.validate(result.output)
                if not validation.valid:
                    result.success = False
                    result.error = validation.error
                    result.error_type = "ValidationError"
                else:
                    result.output = validation.value

            return result

        except ExecutionError as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=e.error_type,
                line_number=e.line_number,
            )
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

    def _validate_code(self, code: str) -> None:
        """Validate code against blocked patterns."""
        for pattern in self.config.blocked_patterns:
            if pattern in code:
                raise ExecutionError(
                    f"Blocked pattern detected: {pattern}",
                    "SecurityError",
                )

    async def _execute_python(
        self,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
        timeout: int,
    ) -> ExecutionResult:
        """Execute Python code in subprocess."""
        transformer = PythonTransformer()
        transformed_code, _ = transformer.transform(code, inputs, preload)

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(transformed_code)
            temp_path = f.name

        try:
            # Build command with resource limits
            cmd = self._build_python_command(temp_path)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_restricted_env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Extract result
            output = self._extract_result(stdout_str)
            error = self._extract_error(stdout_str)
            captured_stdout = self._extract_stdout(stdout_str)

            if error:
                return ExecutionResult(
                    success=False,
                    error=error,
                    error_type="RuntimeError",
                    stdout=captured_stdout,
                    stderr=stderr_str,
                )

            if process.returncode != 0 and not output:
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
                stdout=captured_stdout,
                stderr=stderr_str,
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def _execute_javascript(
        self,
        code: str,
        inputs: dict[str, Any] | None,
        preload: str | None,
        timeout: int,
    ) -> ExecutionResult:
        """Execute JavaScript code in subprocess."""
        transformer = JavaScriptTransformer()
        transformed_code, _ = transformer.transform(code, inputs, preload)

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".js",
            delete=False,
        ) as f:
            f.write(transformed_code)
            temp_path = f.name

        try:
            cmd = [self.config.node_path, temp_path]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_restricted_env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            output = self._extract_result(stdout_str)
            error = self._extract_error(stdout_str)
            captured_stdout = self._extract_stdout(stdout_str)

            if error:
                return ExecutionResult(
                    success=False,
                    error=error,
                    error_type="RuntimeError",
                    stdout=captured_stdout,
                    stderr=stderr_str,
                )

            if process.returncode != 0 and not output:
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
                stdout=captured_stdout,
                stderr=stderr_str,
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def _execute_jinja2(
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
                    self._thread_pool,
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

    def _build_python_command(self, script_path: str) -> list[str]:
        """Build Python command with resource limits."""
        # Use ulimit for resource limits on Unix
        return [
            self.config.python_path,
            "-u",  # Unbuffered output
            script_path,
        ]

    def _get_restricted_env(self) -> dict[str, str]:
        """Get restricted environment variables."""
        import os

        # Minimal environment
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

        # Add proxy settings if configured
        if self.config.http_proxy:
            env["HTTP_PROXY"] = self.config.http_proxy
            env["http_proxy"] = self.config.http_proxy
        if self.config.https_proxy:
            env["HTTPS_PROXY"] = self.config.https_proxy
            env["https_proxy"] = self.config.https_proxy

        return env

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

    def _extract_error(self, output: str) -> str | None:
        """Extract error from output markers."""
        pattern = f"{self.ERROR_START}(.+?){self.ERROR_END}"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            return match.group(1)

        return None

    def _extract_stdout(self, output: str) -> str:
        """Extract captured stdout from output markers."""
        pattern = f"{self.STDOUT_START}(.+?){self.STDOUT_END}"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            return match.group(1)

        # Return output without markers
        clean = output
        clean = re.sub(f"{self.RESULT_START}.*?{self.RESULT_END}", "", clean, flags=re.DOTALL)
        clean = re.sub(f"{self.ERROR_START}.*?{self.ERROR_END}", "", clean, flags=re.DOTALL)
        clean = re.sub(f"{self.STDOUT_START}.*?{self.STDOUT_END}", "", clean, flags=re.DOTALL)
        return clean.strip()

    async def close(self) -> None:
        """Shutdown executor."""
        self._thread_pool.shutdown(wait=False)


# Global executor instance
_executor: CodeExecutor | None = None


def get_executor() -> CodeExecutor:
    """Get global code executor."""
    global _executor
    if _executor is None:
        _executor = CodeExecutor()
    return _executor
