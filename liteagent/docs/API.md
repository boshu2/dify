# LiteAgent API Reference

## Base URL

```
http://localhost:8000/api
```

## Authentication

Currently using API key authentication (optional). Include in header:

```
Authorization: Bearer <api_key>
```

---

## Agents API

### Create Agent

```http
POST /api/agents
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "Support Agent",
  "description": "Customer support assistant",
  "system_prompt": "You are a helpful customer support agent...",
  "provider_id": "uuid-of-llm-provider",
  "datasource_ids": ["uuid-1", "uuid-2"]
}
```

**Response (201):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Support Agent",
  "description": "Customer support assistant",
  "system_prompt": "You are a helpful customer support agent...",
  "provider_id": "uuid-of-llm-provider",
  "provider": {
    "id": "uuid-of-llm-provider",
    "name": "GPT-4o",
    "provider_type": "openai",
    "model_name": "gpt-4o",
    "api_key": "***hidden***",
    "base_url": null,
    "is_active": true,
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:30:00Z"
  },
  "datasources": [
    {
      "id": "uuid-1",
      "name": "FAQ Document",
      "source_type": "file",
      "content": "...",
      "source_path": "faq.md",
      "gitlab_url": null,
      "created_at": "2025-01-15T10:00:00Z",
      "updated_at": "2025-01-15T10:00:00Z"
    }
  ],
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:30:00Z"
}
```

### List Agents

```http
GET /api/agents
```

**Response (200):**
```json
[
  {
    "id": "...",
    "name": "Support Agent",
    "description": "...",
    "system_prompt": "...",
    "provider_id": "...",
    "provider": {...},
    "datasources": [...],
    "created_at": "...",
    "updated_at": "..."
  }
]
```

### Get Agent

```http
GET /api/agents/{agent_id}
```

**Response (200):** Same as create response

**Error (404):**
```json
{
  "detail": "Agent not found"
}
```

### Update Agent

```http
PATCH /api/agents/{agent_id}
Content-Type: application/json
```

**Request Body (all fields optional):**
```json
{
  "name": "Updated Name",
  "description": "Updated description",
  "system_prompt": "Updated prompt...",
  "provider_id": "new-provider-uuid",
  "datasource_ids": ["new-uuid-1", "new-uuid-2"]
}
```

**Response (200):** Updated agent object

### Delete Agent

```http
DELETE /api/agents/{agent_id}
```

**Response (200):**
```json
{
  "status": "deleted"
}
```

### Chat with Agent

```http
POST /api/agents/{agent_id}/chat
Content-Type: application/json
```

**Request Body:**
```json
{
  "message": "Hello, I need help with my order",
  "conversation_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

**Response (200):**
```json
{
  "response": "I'd be happy to help with your order. Could you provide your order number?",
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 25,
    "total_tokens": 175
  }
}
```

---

## Chat API

### Streaming Chat

```http
POST /api/chat/stream
Content-Type: application/json
Accept: text/event-stream
```

**Request Body:**
```json
{
  "agent_id": "default",
  "message": "Hello, how are you?",
  "conversation_history": [],
  "system_prompt": "You are a helpful assistant."
}
```

**Response (200 - Server-Sent Events):**
```
data: {"type": "content", "content": "Hello"}

data: {"type": "content", "content": "! I'm"}

data: {"type": "content", "content": " doing well"}

data: {"type": "tool_call", "tool": "search", "args": {"query": "..."}}

data: {"type": "tool_result", "result": {"data": "..."}}

data: {"type": "done"}
```

**Event Types:**
| Type | Description |
|------|-------------|
| `content` | Text chunk from LLM |
| `tool_call` | Tool invocation request |
| `tool_result` | Tool execution result |
| `done` | Stream complete |
| `error` | Error occurred |

### Non-Streaming Chat

```http
POST /api/chat/complete
Content-Type: application/json
```

**Request Body:**
```json
{
  "agent_id": "default",
  "message": "What is 2+2?",
  "conversation_history": [],
  "system_prompt": "You are a math tutor."
}
```

**Response (200):**
```json
{
  "response": "2 + 2 equals 4.",
  "status": "completed",
  "steps": 2
}
```

### Get Agent State

```http
GET /api/chat/agents/{agent_id}/state
```

**Response (200):**
```json
{
  "agent_id": "agent-123",
  "status": "idle",
  "message": "Agent state info"
}
```

### Chat Health Check

```http
GET /api/chat/health
```

**Response (200):**
```json
{
  "status": "healthy",
  "service": "chat",
  "features": {
    "streaming": true,
    "twelve_factor_agent": true,
    "rag_enabled": false
  }
}
```

---

## DataSources API

### Create DataSource

```http
POST /api/datasources
Content-Type: application/json
```

**Request Body (Text):**
```json
{
  "name": "Product FAQ",
  "source_type": "text",
  "content": "Q: What is your return policy?\nA: 30 days..."
}
```

**Request Body (URL):**
```json
{
  "name": "Company Blog",
  "source_type": "url",
  "source_path": "https://example.com/blog"
}
```

**Request Body (GitLab):**
```json
{
  "name": "Project Docs",
  "source_type": "gitlab",
  "source_path": "project:123/file:README.md@main",
  "gitlab_url": "https://gitlab.example.com",
  "gitlab_token": "glpat-xxx..."
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "name": "Product FAQ",
  "source_type": "text",
  "content": "Q: What is your return policy?\nA: 30 days...",
  "source_path": null,
  "gitlab_url": null,
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-01-15T10:00:00Z"
}
```

### Upload File

```http
POST /api/datasources/upload
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: Binary file content
- `name`: DataSource name

**Supported Extensions:** `.txt`, `.md`, `.json`, `.csv`, `.xml`, `.yaml`, `.yml`, `.py`, `.js`

**Response (201):** DataSource object with file content

### List DataSources

```http
GET /api/datasources
```

**Response (200):** Array of DataSource objects

### Get DataSource

```http
GET /api/datasources/{datasource_id}
```

**Response (200):** DataSource object

### Update DataSource

```http
PATCH /api/datasources/{datasource_id}
Content-Type: application/json
```

**Request Body (all fields optional):**
```json
{
  "name": "Updated Name",
  "content": "Updated content...",
  "source_path": "https://new-url.com",
  "gitlab_token": "new-token"
}
```

**Response (200):** Updated DataSource object

### Delete DataSource

```http
DELETE /api/datasources/{datasource_id}
```

**Response (200):**
```json
{
  "status": "deleted"
}
```

### Refresh DataSource

```http
POST /api/datasources/{datasource_id}/refresh
```

Re-fetches content from URL or GitLab sources.

**Response (200):** Updated DataSource with fresh content

---

## Providers API

### Create Provider

```http
POST /api/providers
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "Production GPT-4o",
  "provider_type": "openai",
  "model_name": "gpt-4o",
  "api_key": "sk-...",
  "base_url": null
}
```

**Provider Types:**
| Type | Description | Required Fields |
|------|-------------|-----------------|
| `openai` | OpenAI API | `api_key` |
| `anthropic` | Anthropic API | `api_key` |
| `ollama` | Local Ollama | `base_url` (default: localhost:11434) |
| `openai_compatible` | Any OpenAI-compatible API | `api_key`, `base_url` |

**Response (201):**
```json
{
  "id": "uuid",
  "name": "Production GPT-4o",
  "provider_type": "openai",
  "model_name": "gpt-4o",
  "api_key": "***hidden***",
  "base_url": null,
  "is_active": true,
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-01-15T10:00:00Z"
}
```

### List Providers

```http
GET /api/providers
GET /api/providers?active_only=true
```

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `active_only` | boolean | false | Filter to active providers only |

**Response (200):** Array of Provider objects

### Get Provider

```http
GET /api/providers/{provider_id}
```

**Response (200):** Provider object

### Update Provider

```http
PATCH /api/providers/{provider_id}
Content-Type: application/json
```

**Request Body (all fields optional):**
```json
{
  "name": "Updated Name",
  "api_key": "new-api-key",
  "base_url": "https://new-url.com",
  "model_name": "gpt-4-turbo",
  "is_active": false
}
```

**Response (200):** Updated Provider object

### Delete Provider

```http
DELETE /api/providers/{provider_id}
```

**Response (200):**
```json
{
  "status": "deleted"
}
```

---

## Metadata API

### List Provider Types

```http
GET /api/meta/provider-types
```

**Response (200):**
```json
{
  "provider_types": [
    {"value": "openai", "label": "OpenAI"},
    {"value": "anthropic", "label": "Anthropic"},
    {"value": "ollama", "label": "Ollama"},
    {"value": "openai_compatible", "label": "OpenAI Compatible"}
  ]
}
```

### List Models for Provider

```http
GET /api/meta/provider-types/{provider_type}/models
```

**Response (200):**
```json
{
  "models": [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1-preview",
    "o1-mini"
  ]
}
```

### List DataSource Types

```http
GET /api/meta/datasource-types
```

**Response (200):**
```json
{
  "datasource_types": [
    {"value": "file", "label": "File"},
    {"value": "url", "label": "URL"},
    {"value": "text", "label": "Text"},
    {"value": "gitlab", "label": "GitLab"}
  ]
}
```

---

## System Endpoints

### Root

```http
GET /
```

**Response (200):**
```json
{
  "name": "LiteAgent",
  "version": "0.1.0",
  "docs": "/docs"
}
```

### Health Check

```http
GET /health
```

**Response (200):**
```json
{
  "status": "healthy"
}
```

### API Documentation

```http
GET /docs      # Swagger UI
GET /redoc     # ReDoc UI
GET /openapi.json  # OpenAPI spec
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing the issue"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid/missing auth |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Schema validation failed |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 502 | Bad Gateway - External service error |

### Domain Exceptions

| Exception | Code | Description |
|-----------|------|-------------|
| `ProviderNotFoundError` | 404 | LLM provider not found |
| `ProviderConfigurationError` | 400 | Invalid provider config |
| `DataSourceNotFoundError` | 404 | DataSource not found |
| `DataSourceFetchError` | 502 | Failed to fetch URL/GitLab |
| `AgentNotFoundError` | 404 | Agent not found |
| `AgentExecutionError` | 500 | Agent execution failed |
| `AuthenticationError` | 401 | Auth failed |
| `RateLimitExceededError` | 429 | Rate limit hit |
| `ValidationError` | 422 | Input validation failed |

---

## Rate Limiting

Rate limits are enforced per API key:

| Limit Type | Default |
|------------|---------|
| Requests per minute | 60 |
| Max concurrent requests | 10 |

**Response Headers:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 55
X-RateLimit-Reset: 1705312800
X-RateLimit-Retry-After: 45
```

---

## GitLab Source Path Formats

For GitLab datasources, `source_path` supports multiple formats:

| Format | Example | Description |
|--------|---------|-------------|
| Single file | `project:123/file:README.md@main` | One file from branch |
| Directory | `project:123/tree:src@main` | Directory listing |
| Issues | `project:123/issues` | All issues |
| Issues filtered | `project:123/issues?state=opened` | Open issues only |
| Merge Requests | `project:123/merge_requests` | All MRs |
| Repository | `project:123/repo@main` | Multiple files (max 50) |

---

## cURL Examples

### Create a provider and agent

```bash
# Create OpenAI provider
curl -X POST http://localhost:8000/api/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GPT-4o",
    "provider_type": "openai",
    "model_name": "gpt-4o",
    "api_key": "sk-..."
  }'

# Create agent with provider
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Assistant",
    "system_prompt": "You are a helpful assistant.",
    "provider_id": "<provider-id-from-above>"
  }'
```

### Streaming chat

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "message": "Tell me a joke",
    "system_prompt": "You are a comedian."
  }'
```

### Upload a file datasource

```bash
curl -X POST http://localhost:8000/api/datasources/upload \
  -F "file=@./document.txt" \
  -F "name=My Document"
```
