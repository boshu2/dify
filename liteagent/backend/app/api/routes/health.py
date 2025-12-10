"""
Health check and diagnostic endpoints.

Provides:
- /health - Overall system health
- /health/live - Kubernetes liveness probe
- /health/ready - Kubernetes readiness probe
- /threads - Thread information
- /db-pool-stat - Database connection pool statistics
"""

import sys
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Comprehensive health check endpoint.

    Returns system health status with details about each component.
    """
    from app.core.health import get_health_registry, HealthStatus

    registry = get_health_registry()
    results = await registry.run_all()
    overall_status = await registry.get_overall_status()

    return {
        "status": overall_status.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [r.to_dict() for r in results],
    }


@router.get("/health/live")
async def liveness_probe(response: Response) -> dict[str, str]:
    """
    Kubernetes liveness probe.

    Returns 200 if the application is running.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe(response: Response) -> dict[str, Any]:
    """
    Kubernetes readiness probe.

    Returns 200 if the application is ready to receive traffic.
    Returns 503 if any critical component is unhealthy.
    """
    from app.core.health import get_health_registry, HealthStatus

    registry = get_health_registry()
    overall_status = await registry.get_overall_status()

    if overall_status == HealthStatus.UNHEALTHY:
        response.status_code = 503

    return {
        "status": "ready" if overall_status != HealthStatus.UNHEALTHY else "not_ready",
        "health_status": overall_status.value,
    }


@router.get("/ping")
async def ping() -> dict[str, str]:
    """Simple ping endpoint for connectivity checks."""
    return {"status": "pong"}


@router.get("/threads")
async def thread_info() -> dict[str, Any]:
    """
    Thread information endpoint.

    Returns information about all running threads.
    Useful for debugging thread leaks and deadlocks.
    """
    threads = []
    for thread in threading.enumerate():
        thread_info = {
            "name": thread.name,
            "ident": thread.ident,
            "daemon": thread.daemon,
            "alive": thread.is_alive(),
        }

        # Try to get native ID (Python 3.8+)
        if hasattr(thread, "native_id"):
            thread_info["native_id"] = thread.native_id

        threads.append(thread_info)

    return {
        "thread_count": threading.active_count(),
        "threads": threads,
        "main_thread": threading.main_thread().name,
    }


@router.get("/db-pool-stat")
async def db_pool_stats() -> dict[str, Any]:
    """
    Database connection pool statistics.

    Returns information about the database connection pool.
    """
    from app.core.database import engine

    pool = engine.pool
    stats: dict[str, Any] = {
        "pool_class": pool.__class__.__name__,
    }

    # Get pool statistics if available
    if hasattr(pool, "size"):
        stats["pool_size"] = pool.size()
    if hasattr(pool, "checkedin"):
        stats["checked_in"] = pool.checkedin()
    if hasattr(pool, "checkedout"):
        stats["checked_out"] = pool.checkedout()
    if hasattr(pool, "overflow"):
        stats["overflow"] = pool.overflow()
    if hasattr(pool, "invalidatedcount"):
        stats["invalidated"] = pool.invalidatedcount()

    # Get timeout settings
    if hasattr(pool, "_pool"):
        stats["timeout"] = getattr(pool, "_timeout", None)
        stats["recycle"] = getattr(pool, "_recycle", None)

    return stats


@router.get("/metrics")
async def metrics_endpoint() -> dict[str, Any]:
    """
    Internal metrics endpoint.

    Returns application metrics in a simple JSON format.
    For Prometheus-compatible metrics, use the /metrics/prometheus endpoint.
    """
    from app.core.logging import metrics

    return {
        "counters": {k: v for k, v in metrics._counters.items()},
        "histograms": {
            k: {
                "count": len(v),
                "min": min(v) if v else 0,
                "max": max(v) if v else 0,
                "avg": sum(v) / len(v) if v else 0,
            }
            for k, v in metrics._histograms.items()
        },
    }


@router.get("/system")
async def system_info() -> dict[str, Any]:
    """
    System information endpoint.

    Returns information about the runtime environment.
    """
    import platform

    import psutil

    process = psutil.Process()
    memory = process.memory_info()

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": psutil.cpu_count(),
        "process": {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_rss_mb": round(memory.rss / (1024 * 1024), 2),
            "memory_vms_mb": round(memory.vms / (1024 * 1024), 2),
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds() if hasattr(process, "num_fds") else None,
            "create_time": datetime.fromtimestamp(
                process.create_time(),
                tz=timezone.utc,
            ).isoformat(),
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "percent_used": psutil.virtual_memory().percent,
        },
    }
