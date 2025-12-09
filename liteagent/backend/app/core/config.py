"""
Application configuration with production-ready settings.

Supports environment-based configuration for all components.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with production defaults."""

    # Application
    app_name: str = "LiteAgent"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # Database
    database_url: str = "sqlite+aiosqlite:///./liteagent.db"

    # Redis Configuration
    redis_mode: Literal["standalone", "sentinel", "cluster"] = "standalone"
    redis_url: str = "redis://localhost:6379/0"
    redis_password: str = ""
    redis_use_ssl: bool = False
    redis_ssl_cert_reqs: str = "required"
    redis_ssl_ca_certs: str | None = None

    # Redis Sentinel (for high availability)
    redis_sentinel_hosts: str = ""  # comma-separated: "host1:26379,host2:26379"
    redis_sentinel_master: str = "mymaster"
    redis_sentinel_password: str = ""
    redis_sentinel_socket_timeout: float = 5.0

    # Redis Cluster
    redis_cluster_nodes: str = ""  # comma-separated: "host1:6379,host2:6379"

    # Redis Connection Pool
    redis_max_connections: int = 50
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0
    redis_retry_on_timeout: bool = True

    # LLM Provider API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_max_concurrent: int = 10

    # Agent Execution
    agent_max_execution_time: int = 300  # 5 minutes
    agent_max_iterations: int = 20
    agent_step_timeout: int = 60

    # Caching TTLs (seconds)
    cache_credentials_ttl: int = 86400  # 24 hours
    cache_embeddings_ttl: int = 600  # 10 minutes
    cache_conversation_ttl: int = 3600  # 1 hour

    # Security
    jwt_secret_key: str = Field(default="change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Observability
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    metrics_enabled: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
