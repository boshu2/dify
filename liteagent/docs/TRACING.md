# LiteAgent Tracing & Observability

This document covers logging, metrics, health checks, and debugging in LiteAgent.

## Overview

LiteAgent provides comprehensive observability through:

1. **Structured Logging** - JSON-formatted logs with context
2. **Metrics Collection** - In-memory counters and histograms
3. **Health Checks** - Kubernetes-compatible probes
4. **Distributed Tracing** - Request correlation across services
5. **Event Streaming** - Real-time agent execution events

## Structured Logging

### Configuration

```python
# app/core/logging.py
from app.core.logging import setup_logging, get_logger

# Configure at startup
setup_logging(
    level="INFO",           # DEBUG, INFO, WARNING, ERROR
    format="json",          # json or text
    include_timestamp=True,
    include_caller=True
)

# Get a logger
logger = get_logger("my_module")
logger.info("Processing request", extra={"user_id": "123", "action": "chat"})
```

### Log Format (JSON)

```json
{
  "timestamp": "2025-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "liteagent.my_module",
  "message": "Processing request",
  "user_id": "123",
  "action": "chat",
  "caller": {
    "file": "my_module.py",
    "line": 42,
    "function": "process"
  }
}
```

### Log Context

Add contextual fields to all logs within a scope:

```python
from app.core.logging import LogContext

async def handle_request(request_id: str, user_id: str):
    with LogContext(request_id=request_id, user_id=user_id):
        logger.info("Starting request")  # Includes request_id, user_id
        await process()
        logger.info("Request complete")  # Includes request_id, user_id
```

### Request Logging

```python
from app.core.logging import RequestLogger

request_logger = RequestLogger()

# At request start
request_logger.log_request_start(
    method="POST",
    path="/api/chat/stream",
    request_id="req-123"
)

# At request end
request_logger.log_request_end(
    method="POST",
    path="/api/chat/stream",
    status=200,
    duration=1.234
)

# On error
request_logger.log_request_error(
    method="POST",
    path="/api/chat/stream",
    error=exception
)
```

### LLM Call Logging

```python
from app.core.logging import LLMCallLogger

llm_logger = LLMCallLogger()

llm_logger.log_call_start(model="gpt-4o", provider="openai")
# ... make LLM call ...
llm_logger.log_call_end(model="gpt-4o", tokens=150, latency=0.8)
```

## Metrics Collection

### Using the Metrics Collector

```python
from app.core.logging import metrics

# Increment counters
metrics.increment("requests_total", labels={"endpoint": "/api/chat"})
metrics.increment("tokens_used", value=150, labels={"model": "gpt-4o"})

# Record histogram values
metrics.record_histogram("request_duration", 1.234, labels={"endpoint": "/api/chat"})
metrics.record_histogram("llm_latency", 0.8, labels={"model": "gpt-4o"})

# Read metrics
count = metrics.get_counter("requests_total", labels={"endpoint": "/api/chat"})
latencies = metrics.get_histogram("llm_latency", labels={"model": "gpt-4o"})
```

### Available Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `requests_total` | Counter | endpoint, method | Total HTTP requests |
| `requests_errors` | Counter | endpoint, status | Error responses |
| `request_duration` | Histogram | endpoint | Request latency (seconds) |
| `llm_calls_total` | Counter | model, provider | Total LLM API calls |
| `llm_tokens_total` | Counter | model, type | Tokens used (input/output) |
| `llm_latency` | Histogram | model | LLM call latency (seconds) |
| `agent_steps` | Counter | agent_id, step_type | Agent execution steps |
| `workflow_executions` | Counter | workflow_id, status | Workflow completions |
| `cache_hits` | Counter | cache_type | Cache hit count |
| `cache_misses` | Counter | cache_type | Cache miss count |

### Exporting Metrics

```python
# Get all metrics as dict
all_metrics = metrics.export()

# Example: Prometheus endpoint
@app.get("/metrics")
async def prometheus_metrics():
    data = metrics.export()
    # Format as Prometheus text format
    lines = []
    for name, values in data["counters"].items():
        lines.append(f"{name} {values['value']}")
    return PlainTextResponse("\n".join(lines))
```

## Health Checks

### Built-in Health Checks

```python
from app.core.health import (
    HealthCheckRegistry,
    DatabaseHealthCheck,
    ExternalServiceHealthCheck,
    DiskSpaceHealthCheck,
    MemoryHealthCheck,
    get_health_registry
)

registry = get_health_registry()

# Database connectivity
registry.register(DatabaseHealthCheck(session_factory=async_session_maker))

# External service (e.g., LLM API)
registry.register(ExternalServiceHealthCheck(
    name="openai",
    url="https://api.openai.com/v1/models",
    timeout=5.0,
    expected_status=200
))

# Disk space (warn if <10% free)
registry.register(DiskSpaceHealthCheck(
    name="disk",
    path="/",
    threshold_percent=10
))

# Memory usage (warn if >90% used)
registry.register(MemoryHealthCheck(
    name="memory",
    threshold_percent=90
))
```

### Health Endpoints

```python
from app.core.health import get_health_registry

@app.get("/health")
async def health():
    """Liveness probe - basic health"""
    return {"status": "healthy"}

@app.get("/health/ready")
async def readiness():
    """Readiness probe - all dependencies"""
    registry = get_health_registry()
    results = await registry.run_all()
    overall = await registry.get_overall_status()

    return {
        "status": overall.value,
        "checks": {
            name: {
                "status": r.status.value,
                "latency_ms": r.latency_ms,
                "details": r.details
            }
            for name, r in results.items()
        }
    }

@app.get("/health/live")
async def liveness():
    """Liveness probe - process alive"""
    return {"status": "alive"}
```

### Custom Health Check

```python
from app.core.health import HealthCheck, HealthCheckResult, HealthStatus
import time

class RedisHealthCheck(HealthCheck):
    def __init__(self, redis_client):
        super().__init__(name="redis")
        self.redis = redis_client

    async def check(self) -> HealthCheckResult:
        start = time.time()
        try:
            await self.redis.ping()
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details={"message": "Redis connected"}
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                details={"error": str(e)}
            )

registry.register(RedisHealthCheck(redis_client))
```

## Distributed Tracing

### Request Correlation

Every request gets a unique ID for tracing:

```python
import uuid
from fastapi import Request

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    with LogContext(request_id=request_id):
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

### Tracing Agent Execution

```python
from app.core.logging import LogContext, get_logger

logger = get_logger("agent")

async def execute_agent(agent_id: str, execution_id: str):
    with LogContext(
        agent_id=agent_id,
        execution_id=execution_id,
        trace_id=str(uuid.uuid4())
    ):
        logger.info("Agent execution started")

        for step in range(max_steps):
            with LogContext(step=step):
                logger.info("Executing step")
                result = await agent.step(state)
                logger.info("Step complete", extra={"status": result.status})

        logger.info("Agent execution complete")
```

### Tracing Workflow Execution

```python
from app.core.logging import LogContext, get_logger

logger = get_logger("workflow")

async def execute_workflow(workflow_id: str, execution_id: str):
    with LogContext(
        workflow_id=workflow_id,
        execution_id=execution_id,
        trace_id=str(uuid.uuid4())
    ):
        logger.info("Workflow started")

        while state.status == WorkflowStatus.RUNNING:
            for node_id in state.current_nodes:
                with LogContext(node_id=node_id, node_type=node.type.value):
                    logger.info("Node execution started")
                    output = await handler.execute(node, state, context)
                    logger.info("Node execution complete", extra={"output_keys": list(output.keys())})

        logger.info("Workflow complete", extra={"final_status": state.status.value})
```

## Real-time Event Streaming

### Publishing Agent Events

```python
from app.core.pubsub import AgentEventPublisher, EventType

publisher = AgentEventPublisher(agent_id="agent-123", execution_id="exec-456")

# Lifecycle events
publisher.started(execution_id="exec-456")
publisher.step({"iteration": 1, "action": "tool_call"})
publisher.completed({"result": "Task completed successfully"})
publisher.failed("Error: API timeout")

# Streaming events
publisher.stream_chunk("Hello")
publisher.stream_chunk(" world")
publisher.stream_tool_call("search", {"query": "weather"})
publisher.stream_tool_result({"temperature": 72})
publisher.stream_end()

# Human-in-the-loop
publisher.human_input_requested(
    request_id="req-789",
    request_type="approval",
    message="Approve this action?"
)
```

### Subscribing to Events

```python
from app.core.pubsub import AgentEventSubscriber

subscriber = AgentEventSubscriber(agent_id="agent-123")

# Subscribe to all events
async for event in subscriber.subscribe(execution_id="exec-456"):
    print(f"Event: {event.event_type} - {event.data}")

# Subscribe to streaming events only
async for event in subscriber.subscribe_streaming(execution_id="exec-456"):
    if event.event_type == EventType.STREAM_CHUNK:
        print(event.data["chunk"], end="")
```

### Event Types

| Event | When | Data |
|-------|------|------|
| `AGENT_STARTED` | Agent execution begins | `{execution_id}` |
| `AGENT_STEP` | Each agent step | `{iteration, action, ...}` |
| `AGENT_COMPLETED` | Successful completion | `{result}` |
| `AGENT_FAILED` | Execution failed | `{error}` |
| `AGENT_PAUSED` | Execution paused | `{reason}` |
| `STREAM_CHUNK` | Text token streamed | `{chunk}` |
| `STREAM_TOOL_CALL` | Tool invocation | `{tool, args}` |
| `STREAM_TOOL_RESULT` | Tool result | `{result}` |
| `STREAM_END` | Stream complete | `{}` |
| `HUMAN_INPUT_REQUESTED` | Waiting for human | `{request_id, type, message}` |
| `HUMAN_INPUT_PROVIDED` | Human responded | `{response}` |

## Debugging

### Debug Mode

Enable debug mode for verbose logging:

```python
# In .env or environment
DEBUG=true
LOG_LEVEL=DEBUG

# Or programmatically
from app.core.config import get_settings
settings = get_settings()
settings.debug = True
```

### Circuit Breaker Stats

```python
from app.core.circuit_breaker import get_circuit_breaker_registry

registry = get_circuit_breaker_registry()

# Get stats for all circuit breakers
stats = registry.get_all_stats()
print(stats)
# {
#     "openai": {
#         "state": "CLOSED",
#         "failure_count": 0,
#         "success_count": 150,
#         "last_failure": null
#     },
#     "anthropic": {
#         "state": "HALF_OPEN",
#         "failure_count": 5,
#         "success_count": 45,
#         "last_failure": "2025-01-15T10:30:00Z"
#     }
# }

# Reset a specific breaker
breaker = registry.get("openai")
breaker.reset()
```

### Rate Limiter Status

```python
from app.core.rate_limiter import get_rate_limiter

limiter = get_rate_limiter()

# Check current status for a client
status = await limiter.get_status(client_key="user-123")
print(f"Remaining: {status.requests_remaining}")
print(f"Reset at: {status.reset_at}")
print(f"Active concurrent: {status.concurrent_active}/{status.concurrent_limit}")
```

### Cache Stats

```python
from app.core.cache import get_llm_cache
from app.core.cache_manager import embedding_cache, conversation_cache

# LLM response cache
llm_cache = get_llm_cache()
stats = llm_cache.get_stats()
print(f"Size: {stats['size']}, Hit rate: {stats['hit_rate']:.2%}")

# Embedding cache hit rate
# (Implement similar get_stats() method)
```

### Session Inspection

```python
from app.core.session_manager import session_manager

# List recent sessions
sessions = await session_manager.list_sessions(
    tenant_id="tenant-1",
    agent_id="agent-123",
    limit=10
)

# Get session details
for session_id in sessions:
    session = await session_manager.get(session_id)
    print(f"Session: {session_id}")
    print(f"  Messages: {len(session.messages)}")
    print(f"  Last activity: {session.last_activity}")
```

### Task Tracking

```python
from app.core.task_tracker import task_tracker

# Get task info
task = await task_tracker.get_task("task-123")
print(f"Status: {task.status}")
print(f"Started: {task.started_at}")
print(f"Owner: {task.owner_id}")

# Check if stopped
is_stopped = await task_tracker.is_stopped("task-123")
```

## Log Aggregation Integration

### Forwarding to External Systems

```python
import logging
from logging.handlers import HTTPHandler

# Forward to Logstash
logstash_handler = HTTPHandler(
    host="logstash.example.com:5000",
    url="/",
    method="POST"
)
logging.getLogger("liteagent").addHandler(logstash_handler)

# Forward to Datadog
from datadog import DogStatsd
statsd = DogStatsd(host="localhost", port=8125)

# In your metrics collector
def send_to_datadog(metric_name: str, value: float, tags: list[str]):
    statsd.gauge(metric_name, value, tags=tags)
```

### Structured Log Fields

Standard fields included in all logs:

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 timestamp |
| `level` | Log level (DEBUG, INFO, WARN, ERROR) |
| `logger` | Logger name (e.g., "liteagent.agents") |
| `message` | Log message |
| `request_id` | HTTP request correlation ID |
| `trace_id` | Distributed trace ID |
| `agent_id` | Agent identifier |
| `execution_id` | Execution identifier |
| `workflow_id` | Workflow identifier |
| `node_id` | Workflow node identifier |
| `user_id` | User identifier |
| `tenant_id` | Tenant identifier |
| `duration` | Operation duration (seconds) |
| `error` | Error details (on failures) |

## Best Practices

### 1. Use Structured Logging

```python
# Good
logger.info("User action completed", extra={
    "user_id": user_id,
    "action": "chat",
    "duration": 1.234
})

# Bad
logger.info(f"User {user_id} completed chat in 1.234s")
```

### 2. Add Context Early

```python
# Add context at the start of request handling
with LogContext(request_id=request_id, user_id=user_id):
    # All logs within this block include request_id and user_id
    await process_request()
```

### 3. Log at Appropriate Levels

```python
logger.debug("Detailed internal state")  # Development only
logger.info("Normal operation events")   # Production visibility
logger.warning("Recoverable issues")     # Needs attention
logger.error("Failures requiring action") # Immediate attention
```

### 4. Include Relevant Data

```python
# On errors, include context for debugging
try:
    result = await llm.chat(messages)
except Exception as e:
    logger.error("LLM call failed", extra={
        "model": model,
        "message_count": len(messages),
        "error_type": type(e).__name__,
        "error": str(e)
    })
    raise
```

### 5. Monitor Key Metrics

- Request latency (p50, p95, p99)
- Error rates by endpoint
- LLM token usage and costs
- Cache hit rates
- Circuit breaker states
- Rate limit triggers
