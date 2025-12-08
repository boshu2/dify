# LiteAgent Architecture

## Overview

LiteAgent is a production-grade platform for building chatbots and 12-factor AI agents. The architecture follows Domain-Driven Design (DDD) and Clean Architecture principles with clear separation of concerns.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                               │
│  (Web UI, CLI, SDK, External Services)                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           API LAYER                                  │
│  FastAPI Routes: /agents, /chat, /datasources, /providers, /meta    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Agents  │ │  Chat   │ │DataSrc  │ │Provider │ │  Meta   │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SERVICE LAYER                                │
│  Business Logic: AgentService, DataSourceService, ProviderService   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  AGENT ENGINE   │    │    RAG PIPELINE     │    │ WORKFLOW ENGINE │
│                 │    │                     │    │                 │
│ ┌─────────────┐ │    │ ┌────────────────┐  │    │ ┌─────────────┐ │
│ │ 12-Factor   │ │    │ │   Chunkers     │  │    │ │   Builder   │ │
│ │   Agent     │ │    │ │ Fixed/Semantic │  │    │ │  (Fluent)   │ │
│ └─────────────┘ │    │ │  /Recursive    │  │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ └────────────────┘  │    │ ┌─────────────┐ │
│ │  RAG Agent  │ │    │ ┌────────────────┐  │    │ │  Reducer    │ │
│ │ (knowledge) │ │    │ │   Embedders    │  │    │ │ (Stateless) │ │
│ └─────────────┘ │    │ │ Nemotron/OpenAI│  │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ └────────────────┘  │    │ ┌─────────────┐ │
│ │   State     │ │    │ ┌────────────────┐  │    │ │  Handlers   │ │
│ │ Management  │ │    │ │ Vector Stores  │  │    │ │ Start/Agent │ │
│ └─────────────┘ │    │ │ PgVector/Memory│  │    │ │ Condition.. │ │
└─────────────────┘    │ └────────────────┘  │    │ └─────────────┘ │
                       │ ┌────────────────┐  │    │ ┌─────────────┐ │
                       │ │   Retrievers   │  │    │ │ Extensions  │ │
                       │ │Vector/BM25/Hyb │  │    │ │Hooks/Custom │ │
                       │ └────────────────┘  │    │ └─────────────┘ │
                       └─────────────────────┘    └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                            │
│                                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Redis   │ │ Database │ │  Celery  │ │   LLM    │ │ External │  │
│  │ Session  │ │SQLAlchemy│ │  Tasks   │ │Providers │ │   APIs   │  │
│  │ PubSub   │ │ Async ORM│ │Background│ │OpenAI/   │ │ GitLab   │  │
│  │ Cache    │ │          │ │          │ │Anthropic │ │ URLs     │  │
│  │ Locks    │ │          │ │          │ │Ollama    │ │          │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent Engine (`app/agents/`)

The 12-factor agent implementation following stateless reducer principles.

| Component | File | Purpose |
|-----------|------|---------|
| Types | `types.py` | AgentStatus, StepType, AgentStep, ToolDefinition, HumanContactRequest |
| State | `state.py` | AgentState with serialization for pause/resume |
| Prompts | `prompts.py` | System prompt generation (Factor 2: Own Your Prompts) |
| Agent | `twelve_factor_agent.py` | Core agent with LLM abstraction |
| RAG Agent | `rag_agent.py` | Agent with integrated knowledge search |

**12-Factor Principles Implemented:**
- Factor 1: Natural Language API (tools defined in plain language)
- Factor 2: Own Your Prompts (centralized prompt management)
- Factor 3: Context Window is RAM (efficient message management)
- Factor 5: Unified State (all state in AgentState)
- Factor 6: Launch/Pause/Resume (serializable state)
- Factor 7: Human in the Loop (request_human_input tool)
- Factor 8: Own Your Control Flow (step-by-step execution)
- Factor 9: Compact Errors (truncated error messages)
- Factor 10: Small Focused Agents (purpose-driven)
- Factor 12: Stateless Reducer (agent as pure function)

### 2. RAG Pipeline (`app/rag/`)

Production-grade retrieval-augmented generation with multiple strategies.

| Component | File | Purpose |
|-----------|------|---------|
| Chunker | `chunker.py` | Text fragmentation (Fixed, Semantic, Recursive) |
| Embeddings | `embeddings.py` | Vector encoding (Nemotron, OpenAI, None) |
| Vector Store | `vector_store.py` | Similarity storage (PgVector, InMemory) |
| Retriever | `retriever.py` | Search strategies (Vector, BM25, Hybrid RRF) |

**Pipeline Flow:**
```
Document → Chunker → Chunks → Embedder → Vectors → VectorStore
                                                         ↓
Query → Embedder → Query Vector → Retriever → Ranked Results
```

### 3. Workflow Engine (`app/workflows/`)

Event-driven workflow orchestration with human-in-the-loop support.

| Component | File | Purpose |
|-----------|------|---------|
| Types | `types.py` | NodeType, WorkflowStatus, Edge, NodeDefinition, NodeExecution |
| State | `state.py` | WorkflowDefinition, WorkflowState |
| Handlers | `handlers.py` | Node executors (Start, End, Agent, Condition, Transform, Human) |
| Reducer | `reducer.py` | Stateless state machine |
| Builder | `builder.py` | Fluent API for workflow construction |
| Engine | `engine.py` | Orchestration (launch, step, run_to_completion) |
| Extensions | `extensions.py` | Custom handlers, hooks, conditions, transformers |

**State Machine:**
```
PENDING → (start) → RUNNING → (step) → RUNNING/COMPLETED/FAILED/WAITING_HUMAN
                        ↓
                    (pause) → PAUSED → (resume) → RUNNING
```

### 4. Core Infrastructure (`app/core/`)

Production-grade infrastructure with 26 modules.

| Category | Modules | Purpose |
|----------|---------|---------|
| Config | `config.py` | Pydantic settings with environment variables |
| Database | `database.py` | Async SQLAlchemy with connection pooling |
| Redis | `redis_client.py` | Standalone/Sentinel/Cluster with SSL |
| Session | `session_manager.py`, `task_tracker.py`, `distributed_lock.py` | Distributed state |
| Resilience | `circuit_breaker.py`, `retry.py`, `rate_limiter.py` | Fault tolerance |
| Caching | `cache.py`, `cache_manager.py` | In-memory and Redis caching |
| Auth | `auth.py`, `encryption.py` | JWT, API keys, Fernet encryption |
| Observability | `logging.py`, `health.py` | Structured logging, health checks |
| Memory | `memory.py` | Conversation memory strategies |
| Tools | `tools.py`, `validation.py` | Tool registry, input validation |
| Streaming | `streaming.py`, `pubsub.py` | SSE responses, Redis pub/sub |
| Extensibility | `registry.py`, `plugins.py`, `prompts.py` | Provider registry, plugins |

### 5. Provider System (`app/providers/`)

Pluggable provider architecture with factory pattern.

**LLM Providers:**
| Provider | Models | Features |
|----------|--------|----------|
| OpenAI | gpt-4o, gpt-4-turbo, gpt-3.5-turbo, o1-* | Chat, streaming, usage tracking |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku | System message handling |
| Ollama | llama3.2, mistral, codellama, phi3 | Local LLM, model discovery |
| OpenAI-Compatible | Azure, vLLM, LiteLLM, LocalAI | Custom base URL, headers |

**DataSource Providers:**
| Provider | Sources | Features |
|----------|---------|----------|
| File | .txt, .md, .json, .csv, .py, .js | Async file I/O |
| URL | HTTP/HTTPS | HTML parsing, BeautifulSoup |
| Text | Inline content | Direct storage |
| GitLab | Files, trees, issues, MRs, repos | API integration |

## Design Patterns

### 1. Stateless Reducer Pattern
```python
# Agent as pure function: state → step() → new_state
state = agent.launch("Hello")
state = await agent.step(state)  # No internal state modified
state = await agent.step(state)  # Each call is independent
```

### 2. Provider Registry Pattern
```python
# Register providers at startup
@llm_registry.register("openai")
class OpenAIProvider(LLMProvider):
    async def chat(self, messages): ...

# Get provider by name
provider = llm_registry.get("openai", api_key="...")
```

### 3. Factory Pattern
```python
provider = LLMProviderFactory.create(
    provider_type=ProviderType.OPENAI,
    api_key="...",
    model_name="gpt-4o"
)
```

### 4. Builder Pattern (Fluent API)
```python
workflow = (WorkflowBuilder("id", "name")
    .add_start()
    .add_agent("agent1", purpose="Process input")
    .add_condition("check", "output == 'success'")
    .add_end()
    .connect("start", "agent1")
    .connect("agent1", "check")
    .connect("check", "end", "condition_result")
    .build())
```

### 5. Strategy Pattern
```python
# Memory strategies
memory = create_token_limited_memory(max_tokens=4000)
memory = create_sliding_window_memory(window_size=10)
memory = create_summary_memory(summarizer=llm_summarize)

# Retrieval strategies
retriever = create_retriever("hybrid", vector_store=vs, embedder=emb)
```

### 6. Hook Pattern
```python
@workflow_hooks.before_node("agent")
async def log_execution(node, state):
    logger.info(f"Executing: {node.id}")
    return state

@workflow_hooks.on_error("*")
async def handle_error(node, state, error):
    notify_admin(error)
    return False  # Don't suppress
```

## Data Models

### SQLAlchemy Models
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   LLMProvider   │     │      Agent      │     │   DataSource    │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id: UUID        │◄────│ provider_id: FK │     │ id: UUID        │
│ name: str       │     │ id: UUID        │     │ name: str       │
│ provider_type   │     │ name: str       │     │ source_type     │
│ model_name      │     │ description     │     │ content         │
│ api_key (enc)   │     │ system_prompt   │     │ source_path     │
│ base_url        │     │ created_at      │     │ gitlab_url      │
│ is_active       │     │ updated_at      │     │ gitlab_token    │
│ created_at      │     └────────┬────────┘     │ created_at      │
│ updated_at      │              │              │ updated_at      │
└─────────────────┘              │              └────────┬────────┘
                                 │                       │
                                 ▼                       │
                    ┌────────────────────────┐          │
                    │   agent_datasources    │◄─────────┘
                    │      (M2M Table)       │
                    ├────────────────────────┤
                    │ agent_id: FK           │
                    │ datasource_id: FK      │
                    └────────────────────────┘
```

## Async Architecture

All I/O operations are asynchronous:

| Layer | Async Library | Purpose |
|-------|---------------|---------|
| API | FastAPI/Starlette | Request handling |
| Database | SQLAlchemy async | ORM operations |
| HTTP | httpx.AsyncClient | External API calls |
| File I/O | aiofiles | File operations |
| Redis | redis.asyncio | Cache, pub/sub |
| LLM SDKs | AsyncOpenAI, AsyncAnthropic | LLM calls |

## Security

| Feature | Implementation |
|---------|----------------|
| API Keys | SHA256 hashing, masked in responses |
| Encryption | Fernet symmetric encryption for secrets |
| JWT | Access/refresh tokens with configurable expiry |
| Rate Limiting | Sliding window + concurrent request limits |
| Input Validation | Prompt injection detection (11 patterns) |
| CORS | Configurable origins |

## Scalability

| Component | Strategy |
|-----------|----------|
| Redis | Standalone → Sentinel → Cluster modes |
| Database | Connection pooling (asyncpg) |
| Background Jobs | Celery with Redis broker |
| Caching | Multi-layer (in-memory + Redis) |
| Circuit Breaker | Automatic failover on API errors |
| Model Fallback | Priority/RoundRobin/Weighted selection |

## Directory Structure

```
liteagent/backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── agents/                 # 12-factor agent engine
│   │   ├── types.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── twelve_factor_agent.py
│   │   └── rag_agent.py
│   ├── api/                    # REST API layer
│   │   └── routes/
│   │       ├── agents.py
│   │       ├── chat.py
│   │       ├── datasources.py
│   │       ├── providers.py
│   │       └── meta.py
│   ├── core/                   # Infrastructure (26 modules)
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── redis_client.py
│   │   ├── session_manager.py
│   │   ├── ... (23 more)
│   │   └── plugins.py
│   ├── models/                 # SQLAlchemy models
│   │   ├── agent.py
│   │   ├── datasource.py
│   │   └── provider.py
│   ├── providers/              # Pluggable providers
│   │   ├── llm/
│   │   │   ├── base.py
│   │   │   ├── openai_provider.py
│   │   │   ├── anthropic_provider.py
│   │   │   ├── ollama_provider.py
│   │   │   └── openai_compatible_provider.py
│   │   └── datasource/
│   │       ├── base.py
│   │       ├── file_provider.py
│   │       ├── url_provider.py
│   │       ├── text_provider.py
│   │       └── gitlab_provider.py
│   ├── rag/                    # RAG pipeline
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── schemas/                # Pydantic schemas
│   │   ├── agent.py
│   │   ├── datasource.py
│   │   └── provider.py
│   ├── services/               # Business logic
│   │   ├── agent_service.py
│   │   ├── datasource_service.py
│   │   └── provider_service.py
│   ├── tasks/                  # Celery background tasks
│   │   ├── datasource_tasks.py
│   │   ├── llm_tasks.py
│   │   └── webhook_tasks.py
│   └── workflows/              # Workflow engine
│       ├── types.py
│       ├── state.py
│       ├── handlers.py
│       ├── reducer.py
│       ├── builder.py
│       ├── engine.py
│       └── extensions.py
├── tests/                      # 760 tests
│   ├── unit/
│   └── integration/
├── pyproject.toml
└── CLAUDE.md
```
