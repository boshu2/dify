"""
Sandbox service API routes.

Matches Dify sandbox API for compatibility.
"""

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.config import SandboxConfig, get_config
from app.executor import CodeExecutor, get_executor
from app.transformers import CodeLanguage

router = APIRouter()


# Request/Response models
class ExecuteRequest(BaseModel):
    """Request to execute code."""

    language: str = Field(..., description="Programming language (python3, javascript, jinja2)")
    code: str = Field(..., description="Code to execute")
    preload: str = Field(default="", description="Preload script")
    enable_network: bool = Field(default=False, description="Enable network access")


class ExecuteData(BaseModel):
    """Execution result data."""

    stdout: str = ""
    error: str = ""


class ExecuteResponse(BaseModel):
    """Response from code execution."""

    code: int = Field(..., description="0 for success, non-zero for error")
    message: str = Field(default="", description="Error message if any")
    data: ExecuteData = Field(default_factory=ExecuteData)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "1.0.0"


class LanguageInfo(BaseModel):
    """Information about a supported language."""

    name: str
    version: str
    available: bool


# Dependency for API key validation
async def verify_api_key(
    x_api_key: str = Header(..., alias="X-Api-Key"),
    config: SandboxConfig = Depends(get_config),
) -> str:
    """Verify API key from header."""
    if x_api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key


# Routes
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@router.get("/v1/sandbox/languages")
async def list_languages(
    _: str = Depends(verify_api_key),
) -> dict[str, list[LanguageInfo]]:
    """List supported programming languages."""
    return {
        "languages": [
            LanguageInfo(name="python3", version="3.11", available=True),
            LanguageInfo(name="javascript", version="18", available=True),
            LanguageInfo(name="jinja2", version="3.1", available=True),
        ]
    }


@router.post("/v1/sandbox/run", response_model=ExecuteResponse)
async def execute_code(
    request: ExecuteRequest,
    _: str = Depends(verify_api_key),
    executor: CodeExecutor = Depends(get_executor),
    config: SandboxConfig = Depends(get_config),
):
    """
    Execute code in sandbox.

    This endpoint matches Dify's sandbox API for compatibility.
    """
    try:
        # Validate language
        try:
            language = CodeLanguage(request.language)
        except ValueError:
            return ExecuteResponse(
                code=400,
                message=f"Unsupported language: {request.language}",
                data=ExecuteData(error=f"Unsupported language: {request.language}"),
            )

        # Decode inputs from preload if present (for backwards compatibility)
        inputs = None
        preload = request.preload if request.preload else None

        # Execute code
        result = await executor.execute(
            language=language,
            code=request.code,
            inputs=inputs,
            preload=preload,
            enable_network=request.enable_network,
            timeout=config.execution_timeout,
        )

        if result.success:
            # Format output for response
            import json

            stdout = result.stdout
            if result.output is not None:
                try:
                    output_json = json.dumps(result.output, default=str)
                    stdout = f"<<RESULT>>{output_json}<<RESULT>>\n{stdout}"
                except Exception:
                    pass

            return ExecuteResponse(
                code=0,
                message="",
                data=ExecuteData(
                    stdout=stdout,
                    error=result.stderr,
                ),
            )
        else:
            return ExecuteResponse(
                code=1,
                message=result.error or "Execution failed",
                data=ExecuteData(
                    stdout=result.stdout,
                    error=result.error or "",
                ),
            )

    except Exception as e:
        return ExecuteResponse(
            code=500,
            message=str(e),
            data=ExecuteData(error=str(e)),
        )


@router.post("/v1/sandbox/run/batch")
async def execute_batch(
    requests: list[ExecuteRequest],
    _: str = Depends(verify_api_key),
    executor: CodeExecutor = Depends(get_executor),
) -> dict[str, list[ExecuteResponse]]:
    """Execute multiple code snippets."""
    import asyncio

    results = await asyncio.gather(
        *[
            execute_code(req, _, executor, get_config())
            for req in requests
        ],
        return_exceptions=True,
    )

    responses = []
    for result in results:
        if isinstance(result, Exception):
            responses.append(
                ExecuteResponse(
                    code=500,
                    message=str(result),
                    data=ExecuteData(error=str(result)),
                )
            )
        else:
            responses.append(result)

    return {"results": responses}


@router.get("/v1/sandbox/config")
async def get_sandbox_config(
    _: str = Depends(verify_api_key),
    config: SandboxConfig = Depends(get_config),
) -> dict[str, Any]:
    """Get sandbox configuration (non-sensitive)."""
    return {
        "max_workers": config.max_workers,
        "execution_timeout": config.execution_timeout,
        "max_memory_mb": config.max_memory_mb,
        "max_output_size": config.max_output_size,
        "enable_network": config.enable_network,
        "max_string_length": config.max_string_length,
        "max_depth": config.max_depth,
        "max_array_length": config.max_array_length,
    }
