"""
Production-grade Redis client with Sentinel/Cluster support.

Provides resilient Redis connections with:
- Standalone, Sentinel, and Cluster modes
- Connection pooling and retry logic
- Graceful fallback for non-critical operations
- SSL/TLS support
"""
import functools
import logging
import ssl
from typing import Any, TypeVar, Callable, ParamSpec

import redis
from redis import Redis, ConnectionPool
from redis.cluster import RedisCluster, ClusterNode
from redis.sentinel import Sentinel

from app.core.config import get_settings

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RedisClientWrapper:
    """
    Deferred initialization wrapper for Redis client.

    Supports standalone, Sentinel, and Cluster modes with
    automatic failover and connection management.
    """

    def __init__(self) -> None:
        self._client: Redis | RedisCluster | None = None
        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize Redis client based on configuration."""
        if self._initialized:
            return

        settings = get_settings()

        try:
            if settings.redis_mode == "sentinel":
                self._client = self._create_sentinel_client(settings)
            elif settings.redis_mode == "cluster":
                self._client = self._create_cluster_client(settings)
            else:
                self._client = self._create_standalone_client(settings)

            # Test connection
            self._client.ping()
            self._initialized = True
            logger.info(
                "Redis client initialized",
                extra={"mode": settings.redis_mode},
            )

        except redis.RedisError as e:
            logger.error(
                "Failed to initialize Redis client",
                extra={"error": str(e), "mode": settings.redis_mode},
            )
            raise

    def _create_standalone_client(self, settings) -> Redis:
        """Create standalone Redis client with connection pooling."""
        pool_kwargs = {
            "password": settings.redis_password or None,
            "max_connections": settings.redis_max_connections,
            "socket_timeout": settings.redis_socket_timeout,
            "socket_connect_timeout": settings.redis_socket_connect_timeout,
            "retry_on_timeout": settings.redis_retry_on_timeout,
        }

        if settings.redis_use_ssl:
            pool_kwargs["ssl"] = True
            pool_kwargs["ssl_cert_reqs"] = settings.redis_ssl_cert_reqs
            if settings.redis_ssl_ca_certs:
                pool_kwargs["ssl_ca_certs"] = settings.redis_ssl_ca_certs

        pool = ConnectionPool.from_url(settings.redis_url, **pool_kwargs)

        return Redis(connection_pool=pool, decode_responses=False)

    def _create_sentinel_client(self, settings) -> Redis:
        """Create Redis client with Sentinel for high availability."""
        if not settings.redis_sentinel_hosts:
            raise ValueError("redis_sentinel_hosts required for sentinel mode")

        sentinel_hosts = [
            tuple(h.strip().split(":"))
            for h in settings.redis_sentinel_hosts.split(",")
        ]
        sentinel_hosts = [
            (host, int(port)) for host, port in sentinel_hosts
        ]

        ssl_context = self._get_ssl_context(settings) if settings.redis_use_ssl else None

        sentinel_kwargs = {
            "socket_timeout": settings.redis_sentinel_socket_timeout,
        }
        if settings.redis_sentinel_password:
            sentinel_kwargs["password"] = settings.redis_sentinel_password
        if ssl_context:
            sentinel_kwargs["ssl"] = True

        sentinel = Sentinel(
            sentinel_hosts,
            sentinel_kwargs=sentinel_kwargs,
        )

        redis_params = {
            "password": settings.redis_password or None,
            "socket_timeout": settings.redis_socket_timeout,
            "retry_on_timeout": settings.redis_retry_on_timeout,
            "decode_responses": False,
        }

        return sentinel.master_for(
            settings.redis_sentinel_master,
            **redis_params,
        )

    def _create_cluster_client(self, settings) -> RedisCluster:
        """Create Redis Cluster client for horizontal scaling."""
        if not settings.redis_cluster_nodes:
            raise ValueError("redis_cluster_nodes required for cluster mode")

        nodes = []
        for node_str in settings.redis_cluster_nodes.split(","):
            host, port = node_str.strip().split(":")
            nodes.append(ClusterNode(host=host, port=int(port)))

        return RedisCluster(
            startup_nodes=nodes,
            password=settings.redis_password or None,
            socket_timeout=settings.redis_socket_timeout,
            retry_on_timeout=settings.redis_retry_on_timeout,
            decode_responses=False,
        )

    def _get_ssl_context(self, settings) -> ssl.SSLContext | None:
        """Create SSL context for secure connections."""
        if not settings.redis_use_ssl:
            return None

        context = ssl.create_default_context()

        if settings.redis_ssl_cert_reqs == "none":
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        elif settings.redis_ssl_cert_reqs == "optional":
            context.verify_mode = ssl.CERT_OPTIONAL
        else:
            context.verify_mode = ssl.CERT_REQUIRED

        if settings.redis_ssl_ca_certs:
            context.load_verify_locations(settings.redis_ssl_ca_certs)

        return context

    @property
    def client(self) -> Redis | RedisCluster:
        """Get initialized Redis client."""
        if not self._initialized:
            self.initialize()
        if self._client is None:
            raise RuntimeError("Redis client not initialized")
        return self._client

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to underlying client."""
        return getattr(self.client, name)


# Global Redis client instance
redis_client = RedisClientWrapper()


def redis_fallback(
    default_return: T | None = None,
    log_error: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """
    Decorator for graceful Redis operation fallback.

    Non-critical operations return default value on Redis errors
    instead of crashing the application.

    Usage:
        @redis_fallback(default_return=[])
        def get_cached_items():
            return redis_client.lrange("items", 0, -1)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            try:
                return func(*args, **kwargs)
            except redis.RedisError as e:
                if log_error:
                    logger.warning(
                        "Redis operation failed, using fallback",
                        extra={
                            "function": func.__name__,
                            "error": str(e),
                            "default": str(default_return),
                        },
                    )
                return default_return
        return wrapper
    return decorator


def ensure_redis() -> bool:
    """Check if Redis is available."""
    try:
        redis_client.ping()
        return True
    except redis.RedisError:
        return False


def get_redis() -> Redis | RedisCluster:
    """Get the Redis client instance."""
    return redis_client.client
