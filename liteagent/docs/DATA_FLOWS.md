# LiteAgent Data Flows

This document traces how data moves through LiteAgent from request to response.

## 1. Chat Flow (Non-Streaming)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHAT FLOW (NON-STREAMING)                           │
└─────────────────────────────────────────────────────────────────────────────┘

Client Request
    │
    │  POST /api/chat/complete
    │  {"message": "Hello", "agent_id": "default", "system_prompt": "..."}
    │
    ▼
┌─────────────────┐
│   FastAPI       │
│   Route Handler │
│   (chat.py)     │
└────────┬────────┘
         │
         │  Validate ChatRequest (Pydantic)
         │
         ▼
┌─────────────────┐
│  Create Agent   │
│  from Config    │
└────────┬────────┘
         │
         │  AgentConfig(agent_id, purpose, tools, llm_client)
         │
         ▼
┌─────────────────┐
│  agent.launch() │
│  (Initialize)   │
└────────┬────────┘
         │
         │  Creates AgentState:
         │  - status: IDLE → RUNNING
         │  - steps: [UserMessage(content)]
         │  - current_iteration: 0
         │
         ▼
┌─────────────────────────────────────────┐
│         agent.run_to_completion()       │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Loop until COMPLETED/FAILED     │   │
│  │                                 │   │
│  │  1. Build LLM messages:         │   │
│  │     - System prompt             │   │
│  │     - Context from state.steps  │   │
│  │                                 │   │
│  │  2. Call LLM:                   │   │
│  │     llm_client.chat(messages)   │──────────┐
│  │                                 │          │
│  │  3. Process response:           │◄─────────┘
│  │     - Tool calls → execute      │     LLM API
│  │     - Message → add to steps    │     Response
│  │                                 │
│  │  4. Update state:               │
│  │     - Add AgentStep             │
│  │     - Increment iteration       │
│  │     - Check terminal condition  │
│  └─────────────────────────────────┘   │
└────────────────────┬────────────────────┘
                     │
                     │  Final AgentState:
                     │  - status: COMPLETED
                     │  - steps: [..., AssistantMessage]
                     │
                     ▼
┌─────────────────┐
│  Extract Last   │
│  Assistant Msg  │
└────────┬────────┘
         │
         │  ChatResponse(response, status, steps)
         │
         ▼
Client Response
```

## 2. Streaming Chat Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STREAMING CHAT FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Client Request (SSE)
    │
    │  POST /api/chat/stream
    │  Accept: text/event-stream
    │
    ▼
┌─────────────────┐
│   FastAPI       │
│   Route Handler │
└────────┬────────┘
         │
         │  Returns StreamingResponse immediately
         │
         ▼
┌─────────────────────────────────────────┐
│         Async Generator                 │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ For each chunk:                 │   │
│  │                                 │   │
│  │  yield f"data: {json}\n\n"      │───────────┐
│  │                                 │           │
│  │  Types:                         │           │
│  │  - {"type": "content", ...}     │           │
│  │  - {"type": "tool_call", ...}   │           │
│  │  - {"type": "tool_result", ...} │           │
│  │  - {"type": "done"}             │           │
│  │  - {"type": "error", ...}       │           │
│  └─────────────────────────────────┘   │       │
└─────────────────────────────────────────┘       │
                                                  │
                                                  ▼
                                           Client receives
                                           SSE events in
                                           real-time
```

## 3. RAG Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAG PIPELINE FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                    INDEXING PHASE
                    ═══════════════

Raw Document
    │
    │  "Long text content..."
    │
    ▼
┌─────────────────┐
│    Chunker      │
│                 │
│ ┌─────────────┐ │
│ │FixedSize    │ │     Chunk(content, index, start, end, metadata)
│ │Semantic     │ │  ─────────────────────────────────────────────►
│ │Recursive    │ │
│ └─────────────┘ │
└─────────────────┘
                           │
                           │  list[Chunk]
                           │
                           ▼
                  ┌─────────────────┐
                  │    Embedder     │
                  │                 │
                  │ ┌─────────────┐ │     HTTP POST
                  │ │Nemotron     │ │  ─────────────►  NVIDIA NIM API
                  │ │OpenAI       │ │                  or OpenAI API
                  │ │None         │ │  ◄─────────────
                  │ └─────────────┘ │     Embeddings
                  └─────────────────┘
                           │
                           │  list[list[float]]  (N x 4096 vectors)
                           │
                           ▼
                  ┌─────────────────┐
                  │  Vector Store   │
                  │                 │
                  │ ┌─────────────┐ │     SQL INSERT
                  │ │PgVector     │ │  ─────────────►  PostgreSQL
                  │ │InMemory     │ │                  + pgvector
                  │ └─────────────┘ │
                  └─────────────────┘
                           │
                           │  Stored: Document(id, content, embedding, metadata)
                           │
                           ▼
                       ✓ INDEXED


                    RETRIEVAL PHASE
                    ════════════════

User Query
    │
    │  "What is the return policy?"
    │
    ▼
┌─────────────────┐
│    Embedder     │
│  (same model)   │
└────────┬────────┘
         │
         │  Query embedding: list[float]
         │
         ▼
┌─────────────────────────────────────────┐
│              Retriever                  │
│                                         │
│  ┌─────────────┐                       │
│  │ Vector      │  Cosine similarity    │
│  │ Retriever   │  search in VectorStore│
│  └─────────────┘                       │
│         ↓                              │
│  ┌─────────────┐                       │
│  │ BM25        │  Term frequency +     │
│  │ Retriever   │  Inverse doc freq     │
│  └─────────────┘                       │
│         ↓                              │
│  ┌─────────────┐                       │
│  │ Hybrid      │  RRF Fusion:          │
│  │ Retriever   │  score = Σ w/(k+rank) │
│  └─────────────┘                       │
└────────────────────┬────────────────────┘
                     │
                     │  RetrievalResult:
                     │  - documents: list[SearchResult]
                     │  - query: str
                     │  - strategy: "hybrid"
                     │  - metadata: {weights, k1, b, ...}
                     │
                     ▼
┌─────────────────┐
│   RAG Agent     │
│                 │
│  Injects docs   │
│  into prompt    │
└────────┬────────┘
         │
         │  Enhanced prompt with context
         │
         ▼
      LLM Call
```

## 4. Workflow Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW EXECUTION FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

WorkflowDefinition
    │
    │  nodes: [start, agent1, condition, human, end]
    │  edges: [start→agent1, agent1→condition, condition→human, human→end]
    │
    ▼
┌─────────────────┐
│ engine.launch() │
│                 │
│  Creates:       │
│  - execution_id │
│  - WorkflowState│
└────────┬────────┘
         │
         │  WorkflowState(status=PENDING, current_nodes=[])
         │
         ▼
┌─────────────────┐
│ engine.start()  │
│                 │
│  Event: "start" │
└────────┬────────┘
         │
         │  reducer.reduce(state, event)
         │
         ▼
┌─────────────────────────────────────────┐
│            WorkflowReducer              │
│                                         │
│  "start" event:                         │
│  1. Set status = RUNNING                │
│  2. Set current_nodes = [start_node]    │
│  3. Initialize variables from input     │
│  4. Execute StartNodeHandler            │
└────────────────────┬────────────────────┘
                     │
                     │  WorkflowState(status=RUNNING, current_nodes=["agent1"])
                     │
                     ▼
┌─────────────────────────────────────────┐
│         engine.step() LOOP              │
│                                         │
│  For each node in current_nodes:        │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ 1. Get NodeHandler for type     │   │
│  │    (AgentNodeHandler)           │   │
│  │                                 │   │
│  │ 2. Execute handler:             │   │
│  │    output = handler.execute(    │───────────┐
│  │      node, state, context       │           │
│  │    )                            │           │  LLM API
│  │                                 │◄──────────┘  calls
│  │ 3. Emit "node_complete" event   │
│  │                                 │
│  │ 4. Reducer updates state:       │
│  │    - Update variables           │
│  │    - Find next nodes via edges  │
│  │    - Evaluate conditions        │
│  │    - Set new current_nodes      │
│  └─────────────────────────────────┘   │
│                                         │
│  Repeat until:                          │
│  - status = COMPLETED (reached END)     │
│  - status = WAITING_HUMAN              │
│  - status = FAILED                     │
│  - status = PAUSED                     │
└────────────────────┬────────────────────┘
                     │
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
COMPLETED      WAITING_HUMAN      FAILED
    │                │                │
    │                │                │
    ▼                ▼                ▼
Return         Wait for           Return
state          human input        error


                HUMAN-IN-THE-LOOP
                ══════════════════

┌─────────────────┐
│ WAITING_HUMAN   │
│                 │
│  HumanNode:     │
│  - prompt       │
│  - options      │
└────────┬────────┘
         │
         │  API call: engine.provide_human_response()
         │
         ▼
┌─────────────────────────────────────────┐
│  "human_response" event:                │
│                                         │
│  1. Store response in variables         │
│  2. Mark human node COMPLETED           │
│  3. Find next nodes                     │
│  4. Set status = RUNNING                │
└────────────────────┬────────────────────┘
                     │
                     │  Continue execution...
                     │
                     ▼
            engine.step() resumes
```

## 5. Agent with DataSource Context Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT WITH DATASOURCE CONTEXT                             │
└─────────────────────────────────────────────────────────────────────────────┘

POST /api/agents/{agent_id}/chat
    │
    │  {"message": "What is your return policy?"}
    │
    ▼
┌─────────────────┐
│ AgentService    │
│   .chat()       │
└────────┬────────┘
         │
         │  1. Load Agent with relationships
         │
         ▼
┌─────────────────────────────────────────┐
│  SELECT * FROM agents                   │
│  JOIN llm_providers ON ...              │
│  JOIN agent_datasources ON ...          │
│  JOIN datasources ON ...                │
│  WHERE agents.id = {agent_id}           │
└────────────────────┬────────────────────┘
                     │
                     │  Agent with:
                     │  - provider: LLMProvider
                     │  - datasources: [DataSource, ...]
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Build Context from DataSources         │
│                                         │
│  for ds in agent.datasources:           │
│      context += f"## {ds.name}\n"       │
│      context += ds.content              │
│      context += "\n---\n"               │
└────────────────────┬────────────────────┘
                     │
                     │  context = "## FAQ\nQ: Return policy?..."
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Inject Context into System Prompt      │
│                                         │
│  system_prompt = f"""                   │
│  {agent.system_prompt}                  │
│                                         │
│  Use the following context:             │
│  {context}                              │
│  """                                    │
└────────────────────┬────────────────────┘
                     │
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Build Messages                         │
│                                         │
│  messages = [                           │
│    {"role": "system", "content": ...},  │
│    ...conversation_history...,          │
│    {"role": "user", "content": msg}     │
│  ]                                      │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Create LLM Provider                    │
│                                         │
│  provider = LLMProviderFactory.create(  │
│    provider_type=agent.provider.type,   │
│    api_key=agent.provider.api_key,      │
│    model_name=agent.provider.model_name │
│  )                                      │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Call LLM                               │
│                                         │
│  response = await provider.chat(msgs)   │──────────────┐
│                                         │              │
│                                         │◄─────────────┘
│                                         │   OpenAI/Anthropic/
│                                         │   Ollama API
└────────────────────┬────────────────────┘
                     │
                     │  LLMResponse(content, model, usage)
                     │
                     ▼
┌─────────────────┐
│ ChatResponse    │
│                 │
│ response: str   │
│ usage: dict     │
└─────────────────┘
```

## 6. DataSource Creation Flow (URL Type)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATASOURCE CREATION (URL TYPE)                            │
└─────────────────────────────────────────────────────────────────────────────┘

POST /api/datasources
    │
    │  {
    │    "name": "Company Docs",
    │    "source_type": "url",
    │    "source_path": "https://example.com/docs"
    │  }
    │
    ▼
┌─────────────────┐
│ Validate Schema │
│ (Pydantic)      │
└────────┬────────┘
         │
         │  DataSourceCreate(name, source_type, source_path)
         │
         ▼
┌─────────────────┐
│ DataSourceService│
│    .create()    │
└────────┬────────┘
         │
         │  Detect source_type == URL
         │
         ▼
┌─────────────────────────────────────────┐
│  Create URLDataSourceProvider           │
│                                         │
│  provider = URLDataSourceProvider()     │
└────────────────────┬────────────────────┘
                     │
                     │  provider.fetch_content(url)
                     │
                     ▼
┌─────────────────────────────────────────┐
│  HTTP GET Request                       │
│                                         │
│  async with httpx.AsyncClient() as c:   │
│      response = await c.get(url)        │──────────────┐
│                                         │              │
│                                         │◄─────────────┘
│                                         │   External
│                                         │   Website
└────────────────────┬────────────────────┘
                     │
                     │  HTML response
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Parse HTML with BeautifulSoup          │
│                                         │
│  1. Remove: <script>, <style>, <nav>,   │
│            <footer>, <header>           │
│                                         │
│  2. Extract: soup.get_text("\n")        │
│                                         │
│  3. Capture: <title> for metadata       │
└────────────────────┬────────────────────┘
                     │
                     │  DataSourceContent(
                     │    content=text,
                     │    source=url,
                     │    metadata={title, content_type}
                     │  )
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Create DataSource Record               │
│                                         │
│  INSERT INTO datasources (              │
│    id, name, source_type,               │
│    content, source_path,                │
│    created_at, updated_at               │
│  )                                      │
└────────────────────┬────────────────────┘
                     │
                     │  Commit transaction
                     │
                     ▼
┌─────────────────┐
│ DataSource      │
│ Response        │
└─────────────────┘
```

## 7. Session Management Flow (Redis)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SESSION MANAGEMENT FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

New Conversation
    │
    ▼
┌─────────────────────────────────────────┐
│  SessionManager.create()                │
│                                         │
│  1. Generate UUID for session_id        │
│  2. Create ConversationSession:         │
│     - session_id                        │
│     - tenant_id                         │
│     - agent_id                          │
│     - messages: []                      │
│     - created_at                        │
│     - last_activity                     │
│                                         │
│  3. Serialize to JSON                   │
│                                         │
│  4. Redis SETEX:                        │
│     key: "session:{id}"                 │
│     ttl: 3600 (1 hour)                  │
│     value: JSON                         │
│                                         │
│  5. Update index:                       │
│     ZADD "sessions:{tenant}:{agent}"    │
│          {timestamp} {session_id}       │
└────────────────────┬────────────────────┘
                     │
                     ▼
                Session Created


Add Message
    │
    │  session_id, message
    │
    ▼
┌─────────────────────────────────────────┐
│  SessionManager.add_message()           │
│                                         │
│  1. Get session from Redis:             │
│     GET "session:{id}"                  │
│                                         │
│  2. Append message to messages[]        │
│                                         │
│  3. Trim to last 100 messages           │
│                                         │
│  4. Update last_activity                │
│                                         │
│  5. Save back to Redis:                 │
│     SETEX "session:{id}" ttl JSON       │
└─────────────────────────────────────────┘


Retrieve Session
    │
    │  session_id
    │
    ▼
┌─────────────────────────────────────────┐
│  SessionManager.get()                   │
│                                         │
│  1. GET "session:{id}"                  │
│                                         │
│  2. If exists:                          │
│     - Deserialize JSON                  │
│     - Refresh TTL: EXPIRE key ttl       │
│     - Return ConversationSession        │
│                                         │
│  3. If not exists:                      │
│     - Return None                       │
└─────────────────────────────────────────┘
```

## 8. Pub/Sub Event Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PUB/SUB EVENT FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Agent Execution
    │
    │
    ▼
┌─────────────────────────────────────────┐
│  AgentEventPublisher                    │
│                                         │
│  Events:                                │
│  - publisher.started(execution_id)      │
│  - publisher.step(data)                 │
│  - publisher.stream_chunk(chunk)        │
│  - publisher.completed(result)          │
│  - publisher.failed(error)              │
└────────────────────┬────────────────────┘
                     │
                     │  PUBLISH to channels:
                     │  - "agent:events:{execution_id}"
                     │  - "agent:events:{agent_id}"
                     │
                     ▼
┌─────────────────────────────────────────┐
│              Redis Pub/Sub              │
│                                         │
│  PUBLISH channel json_event             │
└────────────────────┬────────────────────┘
                     │
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Subscriber 1    │     │ Subscriber 2    │
│ (API endpoint)  │     │ (WebSocket)     │
│                 │     │                 │
│ Subscription:   │     │ Subscription:   │
│ _listen() loop  │     │ _listen() loop  │
│                 │     │                 │
│ for event in    │     │ for event in    │
│   subscription: │     │   subscription: │
│   yield event   │     │   ws.send(event)│
└─────────────────┘     └─────────────────┘


Control Commands
    │
    │  pause/resume/abort
    │
    ▼
┌─────────────────────────────────────────┐
│  CommandChannel                         │
│                                         │
│  send_command():                        │
│  - LPUSH "agent:commands:{exec_id}"     │
│    JSON(command_type, data)             │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Agent checks for commands:             │
│                                         │
│  commands = LRANGE key 0 -1             │
│  DELETE key                             │
│                                         │
│  for cmd in commands:                   │
│      if cmd.type == "pause":            │
│          state.status = PAUSED          │
│      if cmd.type == "abort":            │
│          state.status = FAILED          │
└─────────────────────────────────────────┘
```

## 9. Rate Limiting Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RATE LIMITING FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Incoming Request
    │
    │
    ▼
┌─────────────────────────────────────────┐
│  CombinedRateLimiter.request()          │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ 1. Sliding Window Check         │   │
│  │                                 │   │
│  │    key = "ratelimit:{client}"   │   │
│  │    window = 60 seconds          │   │
│  │                                 │   │
│  │    Redis operations:            │   │
│  │    - ZREMRANGEBYSCORE (cleanup) │   │
│  │    - ZCARD (count requests)     │   │
│  │    - ZADD (record request)      │   │
│  │    - EXPIRE (set TTL)           │   │
│  │                                 │   │
│  │    if count >= max_requests:    │   │
│  │        raise RateLimitExceeded  │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ 2. Concurrent Request Check     │   │
│  │                                 │   │
│  │    key = "concurrent:{client}"  │   │
│  │                                 │   │
│  │    Redis operations:            │   │
│  │    - HLEN (count active)        │   │
│  │    - HSET (register request)    │   │
│  │                                 │   │
│  │    if active >= max_concurrent: │   │
│  │        raise ConcurrentExceeded │   │
│  └─────────────────────────────────┘   │
└────────────────────┬────────────────────┘
                     │
                     │  Request allowed
                     │
                     ▼
             Process Request
                     │
                     ▼
┌─────────────────────────────────────────┐
│  On completion:                         │
│                                         │
│  CombinedRateLimiter.exit_request()     │
│                                         │
│  Redis: HDEL "concurrent:{client}" id   │
└─────────────────────────────────────────┘
```

## 10. Circuit Breaker Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CIRCUIT BREAKER FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                     ┌──────────────────┐
                     │     CLOSED       │
                     │   (Normal Op)    │
                     │                  │
                     │ failures < 5     │
                     └────────┬─────────┘
                              │
                              │ failures >= 5
                              │
                              ▼
                     ┌──────────────────┐
                     │      OPEN        │
                     │   (Blocking)     │
                     │                  │
                     │ All calls fail   │
                     │ immediately with │
                     │ CircuitOpenError │
                     └────────┬─────────┘
                              │
                              │ After timeout (30s)
                              │
                              ▼
                     ┌──────────────────┐
                     │   HALF_OPEN      │
                     │   (Testing)      │
                     │                  │
                     │ Allow 1 request  │
                     └────────┬─────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
          Success                        Failure
               │                             │
               ▼                             ▼
        ┌──────────────┐              ┌──────────────┐
        │   CLOSED     │              │    OPEN      │
        │  (Reset)     │              │  (Re-open)   │
        └──────────────┘              └──────────────┘


Usage in LLM calls:
┌─────────────────────────────────────────┐
│  @circuit_breaker("openai")             │
│  async def call_openai(messages):       │
│      breaker = registry.get("openai")   │
│                                         │
│      if breaker.state == OPEN:          │
│          raise CircuitOpenError()       │
│                                         │
│      try:                               │
│          result = await llm.chat(...)   │
│          breaker._on_success()          │
│          return result                  │
│      except Exception as e:             │
│          breaker._on_failure()          │
│          raise                          │
└─────────────────────────────────────────┘
```
