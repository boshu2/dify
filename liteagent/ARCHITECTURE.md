# LiteAgent Architecture

## Service Architecture

LiteAgent follows a modular microservices architecture inspired by Dify, optimized for simplicity and scalability.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (Browser)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Nginx Gateway (80/443)                      │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐           │
│  │ /api/*  │ /v1/*   │ /files  │ /health │ /*      │           │
│  │   ▼     │   ▼     │    ▼    │    ▼    │   ▼     │           │
│  │  API    │  API    │   API   │   API   │  Web    │           │
│  └─────────┴─────────┴─────────┴─────────┴─────────┘           │
└─────────────────────────────────────────────────────────────────┘
          │                                      │
          ▼                                      ▼
┌──────────────────────┐              ┌──────────────────────┐
│   API Server (8000)  │              │   Web Server (3000)  │
│   ─────────────────  │              │   ─────────────────  │
│   FastAPI            │              │   Next.js (future)   │
│   REST API           │              │   React UI           │
│   WebSocket          │              │   Dashboard          │
│   Auth               │              │                      │
└──────────────────────┘              └──────────────────────┘
          │
          ├──────────────┬──────────────┬──────────────┐
          ▼              ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Sandbox   │  │   Worker    │  │   Redis     │  │  PostgreSQL │
│   (8194)    │  │   (Celery)  │  │   (6379)    │  │   (5432)    │
│   ─────────│  │   ─────────│  │   ─────────│  │   ─────────│
│   Code Exec │  │   Async     │  │   Cache     │  │   Metadata  │
│   Python    │  │   Tasks     │  │   Broker    │  │   Workflows │
│   JS        │  │   Indexing  │  │   Sessions  │  │   Users     │
│   Jinja2    │  │   Workflows │  │   Pub/Sub   │  │   Configs   │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   Qdrant (6333) │
              │   ───────────── │
              │   Vector DB     │
              │   Embeddings    │
              │   RAG Search    │
              └─────────────────┘
```

## Services

### 1. API Server
- **Port**: 8000
- **Technology**: FastAPI + SQLAlchemy + Pydantic
- **Responsibilities**:
  - REST API endpoints
  - Authentication (JWT)
  - Conversation management
  - Workflow orchestration
  - Agent execution
  - RAG pipeline coordination

### 2. Worker (Celery)
- **Technology**: Celery + Redis broker
- **Queues**:
  - `default` - General tasks
  - `indexing` - Document indexing
  - `workflow` - Workflow execution
  - `embedding` - Embedding generation
- **Responsibilities**:
  - Async document processing
  - Background workflow execution
  - Scheduled tasks
  - Long-running operations

### 3. Sandbox
- **Port**: 8194
- **Technology**: FastAPI (isolated container)
- **Responsibilities**:
  - Safe Python code execution
  - JavaScript (Node.js) execution
  - Jinja2 template rendering
  - Resource limits and timeouts
- **Security**:
  - Isolated container
  - No external network (sandbox-internal network)
  - API key authentication
  - Blocked dangerous patterns

### 4. Web (Future)
- **Port**: 3000
- **Technology**: Next.js + React + TypeScript
- **Responsibilities**:
  - Dashboard UI
  - Workflow builder
  - Agent configuration
  - Knowledge base management

### 5. Nginx Gateway
- **Ports**: 80 (HTTP), 443 (HTTPS)
- **Routing**:
  - `/api/*` → API Server
  - `/v1/*` → API Server
  - `/health` → API Server
  - `/*` → Web Server (default)

### 6. PostgreSQL
- **Port**: 5432
- **Data**:
  - Users and authentication
  - Conversations and messages
  - Workflows and configurations
  - Agent definitions
  - Knowledge bases metadata

### 7. Redis
- **Port**: 6379
- **Databases**:
  - DB 0: Cache and sessions
  - DB 1: Celery task broker
- **Usage**:
  - Session storage
  - Rate limiting
  - Task queue
  - Pub/sub for real-time updates

### 8. Qdrant
- **Ports**: 6333 (HTTP), 6334 (gRPC)
- **Data**:
  - Document embeddings
  - Semantic search index
- **Usage**:
  - RAG retrieval
  - Similarity search

## Network Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    default network                       │
│  ┌─────┐  ┌─────┐  ┌──────┐  ┌────────┐  ┌──────┐     │
│  │nginx│  │ api │  │worker│  │postgres│  │redis │     │
│  └─────┘  └─────┘  └──────┘  └────────┘  └──────┘     │
│              │                                          │
│              │ (sandbox calls)                          │
│              ▼                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │            sandbox-internal network               │  │
│  │  ┌───────┐                                       │  │
│  │  │sandbox│  (isolated, no external access)       │  │
│  │  └───────┘                                       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Environment Configuration

### Required Variables

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/liteagent

# Redis
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1

# Sandbox
SANDBOX_MODE=remote
SANDBOX_ENDPOINT=http://sandbox:8194
SANDBOX_API_KEY=liteagent-sandbox

# Vector DB
QDRANT_URL=http://qdrant:6333

# Security
SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## Data Flow

### 1. Chat Request Flow
```
Client → Nginx → API → Agent → LLM Provider
                    ↓
              Save to PostgreSQL
                    ↓
              Return response
```

### 2. RAG Query Flow
```
Client → Nginx → API → Embed query
                    ↓
              Qdrant (similarity search)
                    ↓
              Retrieve documents
                    ↓
              LLM with context
                    ↓
              Return response
```

### 3. Document Indexing Flow
```
Client → Nginx → API → Queue task
                    ↓
              Worker (async)
                    ↓
              Parse document
                    ↓
              Chunk text
                    ↓
              Generate embeddings
                    ↓
              Store in Qdrant
                    ↓
              Update PostgreSQL
```

### 4. Code Execution Flow
```
Workflow Node → API → Sandbox (isolated)
                    ↓
              Execute code
                    ↓
              Validate output
                    ↓
              Return result
```

## Scaling

### Horizontal Scaling
- **API**: Multiple instances behind Nginx
- **Worker**: Multiple Celery workers
- **Sandbox**: Multiple instances (stateless)

### Vertical Scaling
- **PostgreSQL**: Increase resources, add read replicas
- **Redis**: Cluster mode for HA
- **Qdrant**: Distributed mode

## Development vs Production

| Aspect | Development | Production |
|--------|-------------|------------|
| Sandbox | `mode=local` (embedded) | `mode=remote` (service) |
| Database | SQLite or local PostgreSQL | Managed PostgreSQL |
| Redis | Local Redis | Redis Cluster |
| SSL | Self-signed | Let's Encrypt |
| Logging | Debug level | Info level |
| Workers | 1 | Auto-scaled |
