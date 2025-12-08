"""
Health check system for LiteAgent.
Provides liveness and readiness probes for Kubernetes.
"""
import asyncio
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import psutil


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

    @property
    def is_ok(self) -> bool:
        """Check if status is acceptable."""
        return self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_healthy(self) -> bool:
        """Check if result indicates healthy status."""
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status.value,
            "checked_at": self.checked_at.isoformat(),
        }
        if self.message:
            result["message"] = self.message
        if self.error:
            result["error"] = self.error
        if self.latency_ms is not None:
            result["latency_ms"] = self.latency_ms
        if self.details:
            result["details"] = self.details
        return result


class HealthCheck(ABC):
    """Base class for health checks."""

    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""

    def __init__(
        self,
        name: str,
        get_session: Callable,
        timeout_seconds: float = 5.0,
    ):
        super().__init__(name, timeout_seconds)
        self.get_session = get_session

    async def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        start = time.monotonic()

        try:
            session = self.get_session()
            await session.execute("SELECT 1")

            latency = (time.monotonic() - start) * 1000

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection OK",
                latency_ms=latency,
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message="Database connection failed",
                error=str(e),
            )


class ExternalServiceHealthCheck(HealthCheck):
    """Health check for external HTTP services."""

    def __init__(
        self,
        name: str,
        url: str,
        client: Any,
        timeout_seconds: float = 5.0,
        expected_status: int = 200,
    ):
        super().__init__(name, timeout_seconds)
        self.url = url
        self.client = client
        self.expected_status = expected_status

    async def check(self) -> HealthCheckResult:
        """Check external service health."""
        start = time.monotonic()

        try:
            response = await self.client.get(self.url, timeout=self.timeout_seconds)
            latency = (time.monotonic() - start) * 1000

            if response.status_code == self.expected_status:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Service responded with {response.status_code}",
                    latency_ms=latency,
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Unexpected status: {response.status_code}",
                    latency_ms=latency,
                )

        except TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message="Service timed out",
                error="Request timed out",
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message="Service check failed",
                error=str(e),
            )


class DiskSpaceHealthCheck(HealthCheck):
    """Health check for disk space."""

    def __init__(
        self,
        name: str,
        path: str = "/",
        min_free_percent: float = 10.0,
        timeout_seconds: float = 5.0,
    ):
        super().__init__(name, timeout_seconds)
        self.path = path
        self.min_free_percent = min_free_percent

    async def check(self) -> HealthCheckResult:
        """Check disk space."""
        try:
            usage = shutil.disk_usage(self.path)
            free_percent = (usage.free / usage.total) * 100

            details = {
                "total_gb": round(usage.total / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "free_percent": round(free_percent, 2),
            }

            if free_percent >= self.min_free_percent:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Disk space OK: {free_percent:.1f}% free",
                    details=details,
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Low disk space: {free_percent:.1f}% free",
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                error=str(e),
            )


class MemoryHealthCheck(HealthCheck):
    """Health check for memory usage."""

    def __init__(
        self,
        name: str,
        max_used_percent: float = 90.0,
        timeout_seconds: float = 5.0,
    ):
        super().__init__(name, timeout_seconds)
        self.max_used_percent = max_used_percent

    async def check(self) -> HealthCheckResult:
        """Check memory usage."""
        try:
            mem = psutil.virtual_memory()

            details = {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_percent": round(mem.percent, 2),
            }

            if mem.percent <= self.max_used_percent:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Memory OK: {mem.percent:.1f}% used",
                    details=details,
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"High memory usage: {mem.percent:.1f}% used",
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                error=str(e),
            )


class CompositeHealthCheck(HealthCheck):
    """Composite health check that combines multiple checks."""

    def __init__(
        self,
        name: str,
        checks: list[HealthCheck],
        timeout_seconds: float = 10.0,
    ):
        super().__init__(name, timeout_seconds)
        self.checks = checks

    async def check(self) -> HealthCheckResult:
        """Run all checks and combine results."""
        results = await asyncio.gather(
            *[c.check() for c in self.checks],
            return_exceptions=True,
        )

        check_results = []
        overall_status = HealthStatus.HEALTHY

        for result in results:
            if isinstance(result, Exception):
                check_results.append({
                    "name": "unknown",
                    "status": "error",
                    "error": str(result),
                })
                overall_status = HealthStatus.UNHEALTHY
            else:
                check_results.append(result.to_dict())
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    result.status == HealthStatus.DEGRADED
                    and overall_status == HealthStatus.HEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

        return HealthCheckResult(
            name=self.name,
            status=overall_status,
            details={"checks": check_results},
        )


class HealthCheckRegistry:
    """Registry for managing health checks."""

    def __init__(self):
        self.checks: dict[str, HealthCheck] = {}

    def register(self, check: HealthCheck) -> None:
        """Register a health check."""
        self.checks[check.name] = check

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self.checks.pop(name, None)

    async def run_all(self) -> list[HealthCheckResult]:
        """Run all registered checks."""
        results = await asyncio.gather(
            *[check.check() for check in self.checks.values()],
            return_exceptions=True,
        )

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_name = list(self.checks.keys())[i]
                processed_results.append(HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    error=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def get_overall_status(self) -> HealthStatus:
        """Get overall health status."""
        results = await self.run_all()

        if not results:
            return HealthStatus.HEALTHY

        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            return HealthStatus.UNHEALTHY

        if any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY


# Global registry
_health_registry: HealthCheckRegistry | None = None


def get_health_registry() -> HealthCheckRegistry:
    """Get the global health check registry."""
    global _health_registry
    if _health_registry is None:
        _health_registry = HealthCheckRegistry()
    return _health_registry
