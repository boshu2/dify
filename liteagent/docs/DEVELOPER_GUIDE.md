# LiteAgent Developer Guide

Complete guide for developing, extending, and deploying LiteAgent.

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (for sessions, pub/sub, caching)
- PostgreSQL 15+ with pgvector (optional, for vector storage)

### Installation

```bash
cd liteagent/backend

# Install dependencies with uv
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

### Configuration

Create `.env` file:

```bash
# Application
APP_NAME=LiteAgent
DEBUG=true
ENVIRONMENT=development

# Database
DATABASE_URL=sqlite+aiosqlite:///./liteagent.db

# Redis
REDIS_URL=redis://localhost:6379

# LLM Providers (at least one required for real chat)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
NVIDIA_API_KEY=nvapi-...

# Optional: Ollama for local LLMs
OLLAMA_BASE_URL=http://localhost:11434
```

### Run the Server

```bash
# Development with auto-reload
uv run uvicorn app.main:app --reload --port 8000

# Production
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
```

## Development Workflow

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/unit/ -v

# Integration tests only
uv run pytest tests/integration/ -v

# With coverage
uv run pytest tests/ --cov=app --cov-report=html

# Specific test file
uv run pytest tests/unit/test_agent.py -v

# Specific test
uv run pytest tests/unit/test_agent.py::test_agent_launch -v
```

### Linting and Type Checking

```bash
# Lint with ruff
uv run ruff check app/

# Auto-fix lint issues
uv run ruff check --fix app/

# Type checking
uv run mypy app/
```

### Code Style

- **Line length**: 100 characters
- **Imports**: Sorted with isort (via ruff)
- **Quotes**: Double quotes for strings
- **Type hints**: Required on all public functions

## Extending LiteAgent

### Adding a New LLM Provider

1. **Create provider class** in `app/providers/llm/`:

```python
# app/providers/llm/my_provider.py
from app.providers.llm.base import BaseLLMProvider, LLMMessage, LLMResponse

class MyLLMProvider(BaseLLMProvider):
    MODELS = ["my-model-1", "my-model-2"]

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "my-model-1",
        base_url: str | None = None
    ):
        super().__init__(api_key, model_name, base_url)
        self.client = MyLLMClient(api_key=api_key, base_url=base_url)

    async def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        response = await self.client.generate(formatted, model=self.model_name)

        return LLMResponse(
            content=response.text,
            model=self.model_name,
            usage={
                "prompt_tokens": response.input_tokens,
                "completion_tokens": response.output_tokens,
                "total_tokens": response.input_tokens + response.output_tokens
            }
        )

    async def stream_chat(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        formatted = [{"role": m.role, "content": m.content} for m in messages]

        async for chunk in self.client.stream(formatted, model=self.model_name):
            yield chunk.text

    @staticmethod
    def get_available_models() -> list[str]:
        return MyLLMProvider.MODELS
```

2. **Register in factory** (`app/providers/llm/factory.py`):

```python
from app.providers.llm.my_provider import MyLLMProvider

class LLMProviderFactory:
    _providers = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.MY_PROVIDER: MyLLMProvider,  # Add here
    }
```

3. **Add to schema** (`app/schemas/provider.py`):

```python
class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MY_PROVIDER = "my_provider"  # Add here
```

4. **Write tests**:

```python
# tests/unit/test_my_provider.py
import pytest
from app.providers.llm.my_provider import MyLLMProvider

@pytest.mark.asyncio
async def test_my_provider_chat():
    provider = MyLLMProvider(api_key="test-key")
    # ... test implementation
```

### Adding a New DataSource Provider

1. **Create provider class** in `app/providers/datasource/`:

```python
# app/providers/datasource/notion_provider.py
from app.providers.datasource.base import BaseDataSourceProvider, DataSourceContent

class NotionDataSourceProvider(BaseDataSourceProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = NotionClient(api_key=api_key)

    async def fetch_content(self, source: str) -> DataSourceContent:
        # source format: "page:abc123" or "database:xyz789"
        source_type, source_id = source.split(":", 1)

        if source_type == "page":
            content = await self._fetch_page(source_id)
        elif source_type == "database":
            content = await self._fetch_database(source_id)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        return DataSourceContent(
            content=content,
            source=source,
            metadata={"type": source_type, "id": source_id}
        )

    async def validate_source(self, source: str) -> bool:
        try:
            source_type, source_id = source.split(":", 1)
            return source_type in ["page", "database"] and len(source_id) > 0
        except ValueError:
            return False

    async def _fetch_page(self, page_id: str) -> str:
        blocks = await self.client.get_block_children(page_id)
        return self._blocks_to_text(blocks)

    async def _fetch_database(self, database_id: str) -> str:
        rows = await self.client.query_database(database_id)
        return self._rows_to_text(rows)
```

2. **Register in factory** and **add to schema** (similar to LLM provider).

### Adding a Custom Tool

```python
from app.core.tools import tool, ToolRegistry

# Using decorator
@tool(name="weather", description="Get current weather for a location")
async def get_weather(location: str, units: str = "celsius") -> dict:
    """
    Args:
        location: City name or coordinates
        units: Temperature units (celsius or fahrenheit)
    """
    # Implementation
    return {"temperature": 22, "conditions": "sunny", "location": location}

# Or manually
from app.core.tools import FunctionTool

weather_tool = FunctionTool(
    func=get_weather,
    name="weather",
    description="Get current weather for a location"
)

# Register
registry = ToolRegistry()
registry.register(weather_tool)
```

### Adding a Custom Workflow Node

```python
# app/workflows/handlers.py (or create new file)
from app.workflows.handlers import NodeHandler
from app.workflows.types import NodeDefinition
from app.workflows.state import WorkflowState

class HTTPRequestHandler(NodeHandler):
    """Execute HTTP requests within workflows."""

    async def execute(
        self,
        node: NodeDefinition,
        state: WorkflowState,
        context: dict
    ) -> dict:
        config = node.config
        url = config["url"]
        method = config.get("method", "GET")
        headers = config.get("headers", {})
        body = config.get("body")

        # Substitute variables in URL
        for var_name, var_value in state.variables.items():
            url = url.replace(f"${{{var_name}}}", str(var_value))

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body
            )

            return {
                "status_code": response.status_code,
                "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                "headers": dict(response.headers)
            }

# Register the handler
from app.workflows.extensions import register_node_handler

@register_node_handler("http_request", description="Make HTTP requests")
class HTTPRequestHandler(NodeHandler):
    # ... implementation
```

Use in workflow:

```python
workflow = (WorkflowBuilder("api-workflow", "API Workflow")
    .add_start()
    .add_node("fetch_data", NodeType.CUSTOM, config={
        "handler": "http_request",
        "url": "https://api.example.com/data/${user_id}",
        "method": "GET"
    })
    .add_end()
    .connect("start", "fetch_data")
    .connect("fetch_data", "end")
    .build())
```

### Adding Workflow Hooks

```python
from app.workflows.extensions import workflow_hooks

@workflow_hooks.before_node("agent")
async def validate_input(node, state):
    """Validate input before agent execution."""
    input_var = node.config.get("input_variable", "input")
    if input_var not in state.variables:
        raise ValueError(f"Missing required variable: {input_var}")
    return state

@workflow_hooks.after_node("agent")
async def log_output(node, state, output):
    """Log agent output for debugging."""
    logger.info(f"Agent {node.id} output: {list(output.keys())}")
    return output

@workflow_hooks.on_error("*")
async def notify_on_error(node, state, error):
    """Send notification on any node error."""
    await send_alert(f"Workflow error in {node.id}: {error}")
    return False  # Don't suppress the error
```

### Adding Custom Condition Evaluators

```python
from app.workflows.extensions import condition_registry

@condition_registry.register("jmespath:")
def jmespath_evaluator(expr: str, context: dict) -> bool:
    """Evaluate JMESPath expressions."""
    import jmespath
    # expr format: "jmespath:response.status == 'success'"
    path = expr.replace("jmespath:", "")
    return bool(jmespath.search(path, context))

@condition_registry.register("regex:")
def regex_evaluator(expr: str, context: dict) -> bool:
    """Match regex patterns."""
    import re
    # expr format: "regex:variable_name,pattern"
    parts = expr.replace("regex:", "").split(",", 1)
    var_name, pattern = parts[0], parts[1]
    value = context.get(var_name, "")
    return bool(re.match(pattern, str(value)))
```

## Testing Strategies

### Unit Testing Agents

```python
import pytest
from unittest.mock import AsyncMock
from app.agents.twelve_factor_agent import Agent, AgentConfig, LLMClient

class MockLLMClient(LLMClient):
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.call_count = 0

    async def chat(self, messages, tools=None):
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

@pytest.mark.asyncio
async def test_agent_completes_simple_task():
    # Arrange
    mock_client = MockLLMClient([
        {"choices": [{"message": {"content": "Task completed!"}}]}
    ])
    config = AgentConfig(
        agent_id="test-agent",
        purpose="Test agent",
        llm_client=mock_client
    )
    agent = Agent(config)

    # Act
    state = agent.launch("Hello")
    state = await agent.run_to_completion(state)

    # Assert
    assert state.status == AgentStatus.COMPLETED
    assert mock_client.call_count == 1
```

### Unit Testing Workflows

```python
@pytest.mark.asyncio
async def test_workflow_conditional_branch():
    # Arrange
    definition = (WorkflowBuilder("test", "Test")
        .add_start()
        .add_condition("check", "value > 10")
        .add_agent("high", purpose="Handle high value")
        .add_agent("low", purpose="Handle low value")
        .add_end()
        .connect("start", "check")
        .connect("check", "high", "condition_result == true")
        .connect("check", "low", "condition_result == false")
        .connect("high", "end")
        .connect("low", "end")
        .build())

    engine = WorkflowEngine()

    # Act - high value path
    state = engine.launch(definition, {"value": 20})
    state = await engine.run_to_completion(definition, state, context)

    # Assert
    assert "high" in state.node_executions
    assert state.node_executions["high"].status == NodeStatus.COMPLETED
```

### Integration Testing with Database

```python
import pytest
from httpx import AsyncClient
from app.main import app
from app.core.database import init_db, engine

@pytest.fixture
async def client():
    # Setup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    # Teardown
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.mark.asyncio
async def test_create_and_use_agent(client):
    # Create provider
    provider_response = await client.post("/api/providers", json={
        "name": "Test GPT",
        "provider_type": "openai",
        "model_name": "gpt-4o",
        "api_key": "test-key"
    })
    provider_id = provider_response.json()["id"]

    # Create agent
    agent_response = await client.post("/api/agents", json={
        "name": "Test Agent",
        "system_prompt": "You are a test assistant.",
        "provider_id": provider_id
    })
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]

    # Chat with agent (will use mock in tests)
    chat_response = await client.post(f"/api/agents/{agent_id}/chat", json={
        "message": "Hello"
    })
    assert chat_response.status_code == 200
```

## Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY app/ ./app/

# Install dependencies
RUN uv sync --no-dev

# Run
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/liteagent
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis

  db:
    image: pgvector/pgvector:pg15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=liteagent
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  celery:
    build: .
    command: uv run celery -A app.core.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/liteagent
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: liteagent-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: liteagent-api
  template:
    metadata:
      labels:
        app: liteagent-api
    spec:
      containers:
        - name: api
          image: liteagent:latest
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: liteagent-secrets
                  key: database-url
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: liteagent-secrets
                  key: redis-url
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | LiteAgent | Application name |
| `DEBUG` | false | Enable debug mode |
| `ENVIRONMENT` | development | Environment (development, staging, production) |
| `DATABASE_URL` | sqlite... | Database connection string |
| `REDIS_URL` | redis://localhost:6379 | Redis connection URL |
| `REDIS_MODE` | standalone | Redis mode (standalone, sentinel, cluster) |
| `REDIS_SSL` | false | Enable Redis SSL/TLS |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `NVIDIA_API_KEY` | - | NVIDIA NIM API key |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama server URL |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | 60 | Rate limit per client |
| `RATE_LIMIT_MAX_CONCURRENT` | 10 | Max concurrent requests |
| `MAX_EXECUTION_TIME` | 300 | Max agent execution time (seconds) |
| `MAX_ITERATIONS` | 20 | Max agent iterations |
| `STEP_TIMEOUT` | 60 | Single step timeout (seconds) |
| `CACHE_TTL_CREDENTIALS` | 86400 | Credentials cache TTL |
| `CACHE_TTL_EMBEDDINGS` | 600 | Embeddings cache TTL |
| `CACHE_TTL_CONVERSATION` | 3600 | Conversation cache TTL |
| `LOG_LEVEL` | INFO | Logging level |
| `LOG_FORMAT` | json | Log format (json, text) |

### Redis Modes

**Standalone** (default):
```bash
REDIS_URL=redis://localhost:6379
REDIS_MODE=standalone
```

**Sentinel** (high availability):
```bash
REDIS_URL=redis://sentinel1:26379,sentinel2:26379,sentinel3:26379
REDIS_MODE=sentinel
REDIS_SENTINEL_MASTER=mymaster
```

**Cluster** (horizontal scaling):
```bash
REDIS_URL=redis://node1:6379,node2:6379,node3:6379
REDIS_MODE=cluster
```

## Troubleshooting

### Common Issues

**1. "Redis connection failed"**
```bash
# Check Redis is running
redis-cli ping

# Check connection URL
REDIS_URL=redis://localhost:6379
```

**2. "LLM API key invalid"**
```bash
# Verify key is set
echo $OPENAI_API_KEY

# Test key directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**3. "Database migration error"**
```bash
# Reset database (development only!)
rm liteagent.db
uv run python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
```

**4. "Circuit breaker open"**
```python
# Reset circuit breaker
from app.core.circuit_breaker import get_circuit_breaker_registry
registry = get_circuit_breaker_registry()
registry.reset_all()
```

**5. "Rate limit exceeded"**
```bash
# Check current limits
curl http://localhost:8000/api/meta/rate-limit-status \
  -H "Authorization: Bearer <token>"

# Headers show remaining quota
# X-RateLimit-Remaining: 55
# X-RateLimit-Reset: 1705312800
```

### Debug Mode

Enable verbose logging:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

Check logs:
```bash
# Follow logs
tail -f logs/liteagent.log

# Search for errors
grep -i error logs/liteagent.log

# JSON log parsing
cat logs/liteagent.log | jq 'select(.level == "ERROR")'
```

## Contributing

### Code Standards

1. **Type hints required** on all public functions
2. **Docstrings** for all public classes and methods
3. **Tests** for all new functionality
4. **No `Any` types** - use specific types
5. **Async-first** - prefer async over sync

### Pull Request Process

1. Fork and create feature branch
2. Write tests first (TDD)
3. Implement feature
4. Run `make lint` and `make test`
5. Update documentation if needed
6. Submit PR with clear description

### Commit Messages

Follow conventional commits:

```
feat: Add Notion datasource provider
fix: Handle empty response from LLM
docs: Update API reference for streaming
test: Add integration tests for workflows
refactor: Extract tool execution to separate module
```
