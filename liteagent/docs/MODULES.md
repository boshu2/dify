# LiteAgent Module Reference

Complete reference for all 70+ modules in LiteAgent.

## Core Infrastructure (`app/core/`)

### config.py
**Purpose:** Application configuration via environment variables

```python
class Settings(BaseSettings):
    # Application
    app_name: str = "LiteAgent"
    debug: bool = False
    environment: str = "development"

    # Database
    database_url: str = "sqlite+aiosqlite:///./liteagent.db"

    # Redis (standalone/sentinel/cluster)
    redis_url: str = "redis://localhost:6379"
    redis_mode: str = "standalone"  # standalone, sentinel, cluster
    redis_ssl: bool = False
    redis_sentinel_master: str = "mymaster"
    redis_cluster_nodes: list[str] = []

    # LLM Providers
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_max_concurrent: int = 10

    # Agent Execution
    max_execution_time: int = 300
    max_iterations: int = 20
    step_timeout: int = 60

    # Caching TTLs (seconds)
    cache_ttl_credentials: int = 86400
    cache_ttl_embeddings: int = 600
    cache_ttl_conversation: int = 3600

def get_settings() -> Settings:
    """Cached singleton settings instance"""
```

### database.py
**Purpose:** Async SQLAlchemy setup

```python
engine = create_async_engine(settings.database_url)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection for database sessions"""

async def init_db() -> None:
    """Create all tables from metadata"""
```

### redis_client.py
**Purpose:** Production Redis client with multiple modes

```python
class RedisClientWrapper:
    def initialize(self) -> None
    def _create_standalone_client(self) -> Redis
    def _create_sentinel_client(self) -> Redis
    def _create_cluster_client(self) -> RedisCluster
    def _get_ssl_context(self) -> ssl.SSLContext | None
    @property
    def client(self) -> Redis | RedisCluster

redis_client = RedisClientWrapper()  # Singleton

def redis_fallback(default=None) -> Callable  # Decorator
def ensure_redis() -> bool
def get_redis() -> Redis | RedisCluster
```

### session_manager.py
**Purpose:** Conversation session management

```python
@dataclass
class ConversationSession:
    session_id: str
    tenant_id: str
    agent_id: str
    messages: list[dict]
    created_at: datetime
    last_activity: datetime
    metadata: dict

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> ConversationSession

class SessionManager:
    async def create(tenant_id: str, agent_id: str, metadata: dict = None) -> ConversationSession
    async def get(session_id: str) -> ConversationSession | None
    async def add_message(session_id: str, message: dict) -> None
    async def get_messages(session_id: str, limit: int = 50) -> list[dict]
    async def list_sessions(tenant_id: str, agent_id: str, limit: int = 50) -> list[str]
    async def delete(session_id: str) -> bool
```

### task_tracker.py
**Purpose:** Agent task lifecycle tracking

```python
class TaskStatus(Enum):
    PENDING, RUNNING, PAUSED, COMPLETED, FAILED, ABORTED

@dataclass
class TaskInfo:
    task_id: str
    agent_id: str
    status: TaskStatus
    owner_id: str
    started_at: datetime
    metadata: dict

class TaskTracker:
    async def start_task(task_id: str, agent_id: str, owner_id: str) -> TaskInfo
    async def get_task(task_id: str) -> TaskInfo | None
    async def update_status(task_id: str, status: TaskStatus, metadata: dict = None) -> None
    async def complete_task(task_id: str, result: dict = None) -> None
    async def fail_task(task_id: str, error: str) -> None
    async def set_stop_flag(task_id: str, owner_id: str) -> bool
    async def is_stopped(task_id: str) -> bool
    async def clear_task(task_id: str) -> None
```

### distributed_lock.py
**Purpose:** Redis-based distributed locking

```python
class DistributedLock:
    def __init__(self, name: str, timeout: int = 30)
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool
    def release(self) -> bool
    def extend(self, additional_time: int) -> bool
    def __enter__(self) -> DistributedLock
    def __exit__(self, *args) -> None

@contextmanager
def distributed_lock(name: str, timeout: int = 30) -> Generator[DistributedLock, None, None]
```

### rate_limiter.py
**Purpose:** Sliding window + concurrent request limiting

```python
class RateLimitInfo:
    requests_remaining: int
    reset_at: datetime
    concurrent_active: int
    concurrent_limit: int

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60)
    async def check(self, key: str) -> RateLimitInfo
    async def get_status(self, key: str) -> RateLimitInfo

class ConcurrentRequestLimiter:
    def __init__(self, max_concurrent: int)
    async def enter(self, key: str) -> str  # Returns request_id
    async def exit(self, key: str, request_id: str) -> None
    async def get_active_count(self, key: str) -> int
    @asynccontextmanager
    async def request_context(self, key: str) -> AsyncGenerator[str, None]

class CombinedRateLimiter:
    async def check_rate_limit(self, key: str) -> RateLimitInfo
    async def enter_request(self, key: str) -> str
    async def exit_request(self, key: str, request_id: str) -> None
    async def get_status(self, key: str) -> RateLimitInfo
    @asynccontextmanager
    async def request(self, key: str) -> AsyncGenerator[RateLimitInfo, None]

def rate_limit_headers(info: RateLimitInfo) -> dict[str, str]
```

### circuit_breaker.py
**Purpose:** Fault tolerance for external APIs

```python
class CircuitState(Enum):
    CLOSED, OPEN, HALF_OPEN

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 30
    half_open_max_calls: int = 1

class CircuitBreaker:
    @property
    def state(self) -> CircuitState
    @property
    def failure_count(self) -> int
    async def call(self, func: Callable, *args, **kwargs) -> Any
    def get_stats(self) -> dict
    def reset(self) -> None

class CircuitBreakerRegistry:
    def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker
    def get(self, name: str) -> CircuitBreaker | None
    def get_all_stats(self) -> dict[str, dict]
    def reset_all(self) -> None

def circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> Callable
```

### cache.py
**Purpose:** In-memory LRU cache with TTL

```python
@dataclass
class CacheConfig:
    max_size: int = 1000
    default_ttl: int = 300
    compression_threshold: int = 1024

class InMemoryCache:
    def get(self, key: str) -> Any | None
    def set(self, key: str, value: Any, ttl: int = None) -> None
    def delete(self, key: str) -> bool
    def exists(self, key: str) -> bool
    def clear(self) -> None
    def get_stats(self) -> dict

class LLMResponseCache:
    def get_cached_response(self, messages: list[dict], model: str) -> dict | None
    def cache_response(self, messages: list[dict], model: str, response: dict) -> None
    def invalidate(self, messages: list[dict], model: str) -> bool
    def get_stats(self) -> dict
```

### cache_manager.py
**Purpose:** Redis-backed caching layer

```python
class CredentialsCache(BaseCache[dict]):
    async def get_provider_credentials(self, provider_id: str) -> dict | None
    async def set_provider_credentials(self, provider_id: str, credentials: dict) -> None
    async def invalidate_provider(self, provider_id: str) -> None

class EmbeddingCache(BaseCache[list[float]]):
    async def get_embedding(self, text: str, model: str) -> list[float] | None
    async def set_embedding(self, text: str, model: str, embedding: list[float]) -> None

class ConversationCache(BaseCache[dict]):
    async def get_context(self, session_id: str) -> dict | None
    async def set_context(self, session_id: str, context: dict) -> None
    async def append_message(self, session_id: str, message: dict) -> None

class AgentConfigCache(BaseCache[dict]):
    async def get_config(self, agent_id: str) -> dict | None
    async def set_config(self, agent_id: str, config: dict) -> None
    async def invalidate(self, agent_id: str) -> None

async def get_cached_or_compute(
    cache: BaseCache,
    key: str,
    compute_func: Callable[[], Awaitable[T]],
    ttl: int = None
) -> T
```

### auth.py
**Purpose:** JWT and API key authentication

```python
@dataclass
class User:
    id: str
    email: str
    roles: list[str]
    def has_role(self, role: str) -> bool

class JWTHandler:
    def create_access_token(self, user: User) -> str
    def create_refresh_token(self, user: User) -> str
    def decode_token(self, token: str) -> TokenPayload

class APIKeyAuth:
    def verify(self, api_key: str, stored_hash: str) -> bool
    def validate(self, api_key: str, stored_hash: str) -> None

class AuthMiddleware:
    def should_skip_auth(self, path: str) -> bool
    async def authenticate(self, authorization: str) -> User

def generate_api_key(prefix: str = "sk") -> str
def hash_api_key(api_key: str) -> str
def verify_api_key(api_key: str, stored_hash: str) -> bool
```

### encryption.py
**Purpose:** Symmetric encryption for secrets

```python
class EncryptionService:
    def __init__(self, key: bytes = None)
    def encrypt(self, plaintext: str) -> str  # Returns "enc:v1:..."
    def decrypt(self, ciphertext: str) -> str
    def is_encrypted(self, value: str) -> bool
    def get_key(self) -> bytes

def get_encryption_service() -> EncryptionService
```

### exceptions.py
**Purpose:** Domain-specific exceptions

```python
class LiteAgentException(Exception):
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    def to_http_exception(self) -> HTTPException

class ProviderNotFoundError(LiteAgentException):  # 404
class ProviderConfigurationError(LiteAgentException):  # 400
class DataSourceNotFoundError(LiteAgentException):  # 404
class DataSourceFetchError(LiteAgentException):  # 502
class AgentNotFoundError(LiteAgentException):  # 404
class AgentExecutionError(LiteAgentException):  # 500
class AuthenticationError(LiteAgentException):  # 401
class RateLimitExceededError(LiteAgentException):  # 429
class ValidationError(LiteAgentException):  # 422
```

### health.py
**Purpose:** Kubernetes health checks

```python
class HealthStatus(Enum):
    HEALTHY, UNHEALTHY, DEGRADED

@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    latency_ms: float
    details: dict

class DatabaseHealthCheck(HealthCheck):
    async def check(self) -> HealthCheckResult

class ExternalServiceHealthCheck(HealthCheck):
    async def check(self) -> HealthCheckResult

class DiskSpaceHealthCheck(HealthCheck):
    async def check(self) -> HealthCheckResult

class MemoryHealthCheck(HealthCheck):
    async def check(self) -> HealthCheckResult

class HealthCheckRegistry:
    def register(self, check: HealthCheck) -> None
    def unregister(self, name: str) -> None
    async def run_all(self) -> dict[str, HealthCheckResult]
    async def get_overall_status(self) -> HealthStatus
```

### logging.py
**Purpose:** Structured logging with metrics

```python
class JSONFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str

class LogContext:
    def __enter__(self) -> LogContext
    def __exit__(self, *args) -> None

class RequestLogger:
    def log_request_start(self, method: str, path: str, request_id: str) -> None
    def log_request_end(self, method: str, path: str, status: int, duration: float) -> None
    def log_request_error(self, method: str, path: str, error: Exception) -> None

class LLMCallLogger:
    def log_call_start(self, model: str, provider: str) -> None
    def log_call_end(self, model: str, tokens: int, latency: float) -> None
    def log_call_error(self, model: str, error: Exception) -> None

class MetricsCollector:
    def increment(self, name: str, value: int = 1, labels: dict = None) -> None
    def get_counter(self, name: str, labels: dict = None) -> int
    def record_histogram(self, name: str, value: float, labels: dict = None) -> None
    def get_histogram(self, name: str, labels: dict = None) -> list[float]
    def reset(self) -> None

metrics = MetricsCollector()
```

### memory.py
**Purpose:** Conversation memory strategies

```python
class MessageRole(Enum):
    SYSTEM, USER, ASSISTANT, TOOL

@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: dict
    def to_dict(self) -> dict

class ConversationHistory:
    def add_message(self, role: MessageRole, content: str) -> None
    def add_user_message(self, content: str) -> None
    def add_assistant_message(self, content: str) -> None
    def set_system_message(self, content: str) -> None
    def get_messages(self) -> list[Message]
    def to_list(self) -> list[dict]
    def clear(self) -> None

class SlidingWindowMemory(MemoryStrategy):
    """Keep last N messages"""
    def apply(self, history: ConversationHistory) -> ConversationHistory

class TokenWindowMemory(MemoryStrategy):
    """Keep messages within token limit"""
    def apply(self, history: ConversationHistory) -> ConversationHistory

class SummaryMemory(MemoryStrategy):
    """Summarize older messages"""
    def apply(self, history: ConversationHistory) -> ConversationHistory

class ConversationMemory:
    def __init__(self, strategy: MemoryStrategy = None)
    def add_message(self, role: MessageRole, content: str) -> None
    def get_messages(self) -> list[dict]
    def clear(self) -> None
```

### model_fallback.py
**Purpose:** LLM model fallback chain

```python
class FallbackReason(Enum):
    ERROR, TIMEOUT, RATE_LIMIT, CAPACITY, UNAVAILABLE, CIRCUIT_OPEN

class SelectionStrategy(Enum):
    PRIORITY, ROUND_ROBIN, RANDOM, WEIGHTED, LEAST_ERRORS

@dataclass
class ModelConfig:
    name: str
    provider: str
    priority: int = 0
    weight: float = 1.0
    enabled: bool = True

class ModelFallbackChain:
    def __init__(self, models: list[ModelConfig], strategy: SelectionStrategy = PRIORITY)
    def get_ordered_models(self) -> list[ModelConfig]
    def get_status(self, model_name: str) -> ModelStatus
    async def execute(self, func: Callable, *args, **kwargs) -> FallbackResult
    def add_model(self, config: ModelConfig) -> None
    def remove_model(self, model_name: str) -> None
    def enable_model(self, model_name: str) -> None
    def disable_model(self, model_name: str) -> None
```

### pubsub.py
**Purpose:** Redis Pub/Sub for real-time events

```python
class EventType(Enum):
    AGENT_STARTED, AGENT_STEP, AGENT_COMPLETED, AGENT_FAILED, AGENT_PAUSED
    STREAM_CHUNK, STREAM_TOOL_CALL, STREAM_TOOL_RESULT, STREAM_END
    CONTROL_PAUSE, CONTROL_RESUME, CONTROL_ABORT
    HUMAN_INPUT_REQUESTED, HUMAN_INPUT_PROVIDED

@dataclass
class AgentEvent:
    event_type: EventType
    execution_id: str
    agent_id: str
    data: dict
    timestamp: datetime
    def to_json(self) -> str
    @classmethod
    def from_json(cls, json_str: str) -> AgentEvent

class Topic:
    def publish(self, event: AgentEvent) -> None
    def subscribe(self) -> Subscription

class Subscription:
    def receive(self, timeout: float = None) -> AgentEvent | None
    def __iter__(self) -> Iterator[AgentEvent]
    def close(self) -> None

class AgentEventPublisher:
    def started(self, execution_id: str) -> None
    def step(self, data: dict) -> None
    def completed(self, result: dict) -> None
    def failed(self, error: str) -> None
    def stream_chunk(self, chunk: str) -> None

class CommandChannel:
    def send_command(self, command_type: str, data: dict = None) -> None
    def fetch_commands(self) -> list[dict]
    def pause(self) -> None
    def resume(self) -> None
    def abort(self) -> None
```

### retry.py
**Purpose:** Exponential backoff retry

```python
@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

class RetryError(Exception):
    """Raised when all retries are exhausted"""

async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    config: RetryConfig = None,
    on_retry: Callable[[int, Exception], None] = None,
) -> T

def retry(config: RetryConfig = None) -> Callable  # Decorator
```

### streaming.py
**Purpose:** Server-Sent Events support

```python
class StreamEventType(Enum):
    TEXT, METADATA, ERROR, DONE

@dataclass
class StreamEvent:
    event_type: StreamEventType
    data: Any
    id: str = None

class StreamingResponse(StarletteResponse):
    def __init__(self, generator: AsyncGenerator[StreamEvent, None])

def format_sse_event(event: StreamEvent) -> str
def parse_sse_event(text: str) -> StreamEvent
```

### token_counter.py
**Purpose:** Token counting and cost estimation

```python
@dataclass
class ModelPricing:
    input_per_1k: float
    output_per_1k: float

class TokenCounter:
    def __init__(self, model: str)
    def count(self, text: str) -> int
    def count_messages(self, messages: list[dict]) -> int

def count_tokens(text: str, model: str = "gpt-4") -> int
def count_message_tokens(messages: list[dict], model: str = "gpt-4") -> int
def get_model_pricing(model: str) -> ModelPricing
def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float
def check_context_limit(messages: list[dict], model: str) -> tuple[bool, int, int]
```

### tools.py
**Purpose:** Tool/function calling system

```python
class ParameterType(Enum):
    STRING, NUMBER, INTEGER, BOOLEAN, ARRAY, OBJECT

@dataclass
class ToolParameter:
    name: str
    type: ParameterType
    description: str
    required: bool = True
    enum: list[str] = None
    def to_schema(self) -> dict

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter]
    def to_openai_format(self) -> dict
    def to_anthropic_format(self) -> dict

class Tool(ABC):
    @abstractmethod
    def get_definition(self) -> ToolDefinition
    @abstractmethod
    async def execute(self, **kwargs) -> Any

class FunctionTool(Tool):
    def __init__(self, func: Callable, name: str = None, description: str = None)

class ToolRegistry:
    def register(self, tool: Tool) -> None
    def register_function(self, func: Callable, name: str = None) -> None
    def get(self, name: str) -> Tool | None
    def get_all(self) -> list[Tool]
    def to_openai_format(self) -> list[dict]

class ToolExecutor:
    def __init__(self, registry: ToolRegistry)
    async def execute(self, tool_call: ToolCall) -> ToolResult
    async def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]

def tool(name: str = None, description: str = None) -> Callable  # Decorator
```

### validation.py
**Purpose:** Input validation and sanitization

```python
class ContentValidator:
    INJECTION_PATTERNS: list[str]  # 11 patterns
    DANGEROUS_PATTERNS: list[str]  # XSS, JavaScript

    def validate(self, content: str) -> ValidationResult

class MessageValidator:
    def validate_message(self, message: dict) -> ValidationResult
    def validate_messages(self, messages: list[dict]) -> ValidationResult

class RequestValidator:
    def validate_chat_request(self, request: dict) -> ValidationResult

def sanitize_string(text: str, max_length: int = None) -> str
def validate_model_name(model: str) -> bool
def check_prompt_injection(content: str) -> bool
```

### registry.py
**Purpose:** Provider registry pattern

```python
class ProviderRegistry[T]:
    def register(self, name: str, description: str = "") -> Callable[[type[T]], type[T]]
    def register_factory(self, name: str, factory: Callable[..., T]) -> None
    def register_instance(self, name: str, instance: T) -> None
    def get(self, name: str = None, **config) -> T
    def get_class(self, name: str) -> type[T]
    def list_providers(self) -> list[str]
    def get_metadata(self, name: str) -> ProviderMetadata
    def get_default(self) -> str | None
    def set_default(self, name: str) -> None
    def has(self, name: str) -> bool
    def unregister(self, name: str) -> None
    def clear(self) -> None

# Global registries
llm_registry: ProviderRegistry[LLMProvider]
embedding_registry: ProviderRegistry[EmbeddingProvider]
vector_store_registry: ProviderRegistry[VectorStoreProvider]
storage_registry: ProviderRegistry[StorageProvider]
chunker_registry: ProviderRegistry[ChunkerProvider]
retriever_registry: ProviderRegistry[RetrieverProvider]
```

### plugins.py
**Purpose:** Auto-registration of providers

```python
def load_all_plugins() -> None:
    """Bootstrap all provider registrations (idempotent)"""

def get_provider_summary() -> dict[str, list[str]]:
    """Returns all registered providers by category"""
```

### prompts.py
**Purpose:** Prompt template system

```python
class TemplateFormat(Enum):
    PYTHON, JINJA2, FSTRING

@dataclass
class PromptVariable:
    name: str
    description: str
    required: bool = True
    default: Any = None
    def validate(self, value: Any) -> bool

class PromptTemplate:
    def __init__(self, template: str, variables: list[PromptVariable] = None)
    def get_variable_names(self) -> list[str]
    def validate_variables(self, values: dict) -> ValidationResult
    def render(self, **values) -> str
    def __call__(self, **values) -> str

class PromptBuilder:
    def add(self, text: str) -> PromptBuilder
    def add_line(self, text: str) -> PromptBuilder
    def add_section(self, title: str, content: str) -> PromptBuilder
    def add_if(self, condition: bool, text: str) -> PromptBuilder
    def set_variable(self, name: str, value: Any) -> PromptBuilder
    def build(self) -> str
    def clear(self) -> PromptBuilder

class PromptRegistry:
    def register(self, name: str, template: PromptTemplate, version: str = "1.0") -> None
    def get(self, name: str, version: str = None) -> PromptTemplate
    def list_templates(self) -> list[str]
```

---

## Agents (`app/agents/`)

### types.py
```python
class AgentStatus(str, Enum):
    IDLE, RUNNING, PAUSED, WAITING_HUMAN, COMPLETED, FAILED

class StepType(str, Enum):
    USER_MESSAGE, ASSISTANT_MESSAGE, TOOL_CALL, TOOL_RESULT
    HUMAN_REQUEST, HUMAN_RESPONSE, ERROR

@dataclass
class AgentStep:
    step_type: StepType
    content: Any
    timestamp: datetime
    metadata: dict
    def to_dict(self) -> dict

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    handler: Callable | None = None
    requires_human_approval: bool = False
    def to_openai_format(self) -> dict

@dataclass
class HumanContactRequest:
    request_id: str
    request_type: str  # "approval", "input", "clarification"
    message: str
    context: dict
    timeout_seconds: int = 3600
```

### state.py
```python
@dataclass
class AgentState:
    agent_id: str
    status: AgentStatus = IDLE
    steps: list[AgentStep] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 10
    created_at: datetime
    metadata: dict

    def add_step(self, step: AgentStep) -> None
    def get_context_messages(self) -> list[dict]
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> AgentState
    def compute_hash(self) -> str
```

### prompts.py
```python
class AgentPrompts:
    @staticmethod
    def system_prompt(agent_purpose: str, tools: list[ToolDefinition]) -> str

    @staticmethod
    def error_context(error: str, tool_name: str) -> str
```

### twelve_factor_agent.py
```python
class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], tools: list[dict] = None) -> dict

@dataclass
class AgentConfig:
    agent_id: str
    purpose: str
    tools: list[ToolDefinition] = field(default_factory=list)
    max_iterations: int = 10
    llm_client: LLMClient | None = None

class Agent:
    def __init__(self, config: AgentConfig)
    async def step(self, state: AgentState) -> AgentState
    def launch(self, user_message: str) -> AgentState
    def pause(self, state: AgentState) -> AgentState
    def resume(self, state: AgentState) -> AgentState
    def provide_human_response(self, state: AgentState, response: str, approved: bool = True) -> AgentState
    async def run_to_completion(self, state: AgentState, human_callback: Callable = None) -> AgentState
```

### rag_agent.py
```python
@dataclass
class RAGAgentConfig:
    agent_id: str
    purpose: str
    retriever: Retriever
    llm_client: LLMClient | None = None
    tools: list[ToolDefinition] = field(default_factory=list)
    max_iterations: int = 10
    top_k: int = 5
    include_sources: bool = True

class RAGAgent:
    def __init__(self, config: RAGAgentConfig)
    def launch(self, user_message: str) -> AgentState
    async def step(self, state: AgentState) -> AgentState
    async def run_to_completion(self, state: AgentState, human_callback: Callable = None) -> AgentState

class RAGPipelineBuilder:
    def with_embedder(self, embedder) -> RAGPipelineBuilder
    def with_vector_store(self, vector_store) -> RAGPipelineBuilder
    def with_documents(self, documents: list[Document]) -> RAGPipelineBuilder
    async def build_retriever(self, strategy: str = "hybrid") -> Retriever

async def create_rag_agent(
    agent_id: str,
    purpose: str,
    retriever: Retriever,
    llm_client: LLMClient = None,
    tools: list[ToolDefinition] = None,
    **kwargs
) -> RAGAgent
```

---

## Workflows (`app/workflows/`)

### types.py
```python
class NodeType(str, Enum):
    START, END, AGENT, CONDITION, PARALLEL, MERGE, LOOP, HUMAN, TRANSFORM, WAIT

class WorkflowStatus(str, Enum):
    PENDING, RUNNING, PAUSED, WAITING_HUMAN, WAITING_EVENT, COMPLETED, FAILED

class NodeStatus(str, Enum):
    PENDING, RUNNING, COMPLETED, FAILED, SKIPPED

@dataclass
class Edge:
    source: str
    target: str
    condition: str | None = None

@dataclass
class NodeDefinition:
    id: str
    type: NodeType
    config: dict
    def to_dict(self) -> dict

@dataclass
class NodeExecution:
    node_id: str
    status: NodeStatus
    started_at: datetime | None
    completed_at: datetime | None
    input_data: dict
    output_data: dict
    error: str | None
    def to_dict(self) -> dict

@dataclass
class WorkflowEvent:
    event_type: str
    node_id: str | None
    data: dict
    timestamp: datetime
```

### state.py
```python
@dataclass
class WorkflowDefinition:
    id: str
    name: str
    nodes: list[NodeDefinition]
    edges: list[Edge]
    metadata: dict

    def get_node(self, node_id: str) -> NodeDefinition | None
    def get_outgoing_edges(self, node_id: str) -> list[Edge]
    def get_incoming_edges(self, node_id: str) -> list[Edge]
    def get_start_node(self) -> NodeDefinition | None
    def to_dict(self) -> dict

@dataclass
class WorkflowState:
    workflow_id: str
    execution_id: str
    status: WorkflowStatus
    current_nodes: list[str]
    node_executions: dict[str, NodeExecution]
    variables: dict
    created_at: datetime
    updated_at: datetime
    metadata: dict

    def get_node_execution(self, node_id: str) -> NodeExecution
    def set_variable(self, name: str, value: Any) -> None
    def get_variable(self, name: str, default: Any = None) -> Any
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> WorkflowState
    def compute_hash(self) -> str
```

### handlers.py
```python
class NodeHandler(ABC):
    @abstractmethod
    async def execute(self, node: NodeDefinition, state: WorkflowState, context: dict) -> dict

class StartNodeHandler(NodeHandler): ...
class EndNodeHandler(NodeHandler): ...
class AgentNodeHandler(NodeHandler): ...
class ConditionNodeHandler(NodeHandler): ...
class TransformNodeHandler(NodeHandler): ...
class HumanNodeHandler(NodeHandler): ...
```

### reducer.py
```python
class WorkflowReducer:
    def register_handler(self, node_type: NodeType, handler: NodeHandler) -> None
    async def reduce(self, definition: WorkflowDefinition, state: WorkflowState, event: WorkflowEvent, context: dict = None) -> WorkflowState
    async def execute_node(self, definition: WorkflowDefinition, state: WorkflowState, node_id: str, context: dict) -> dict
```

### builder.py
```python
class WorkflowBuilder:
    def __init__(self, workflow_id: str, name: str)
    def add_start(self, node_id: str = "start") -> WorkflowBuilder
    def add_end(self, node_id: str = "end") -> WorkflowBuilder
    def add_agent(self, node_id: str, agent_id: str = None, purpose: str = "", input_variable: str = "input", output_variable: str = "output") -> WorkflowBuilder
    def add_condition(self, node_id: str, condition: str) -> WorkflowBuilder
    def add_transform(self, node_id: str, transform: str, input_variable: str, output_variable: str, template: str = None) -> WorkflowBuilder
    def add_human(self, node_id: str, prompt: str, options: list[str] = None) -> WorkflowBuilder
    def connect(self, source: str, target: str, condition: str = None) -> WorkflowBuilder
    def build(self) -> WorkflowDefinition
```

### engine.py
```python
class WorkflowEngine:
    def launch(self, definition: WorkflowDefinition, input_data: dict = None) -> WorkflowState
    async def start(self, definition: WorkflowDefinition, state: WorkflowState, input_data: dict = None, context: dict = None) -> WorkflowState
    async def step(self, definition: WorkflowDefinition, state: WorkflowState, context: dict = None) -> WorkflowState
    async def run_to_completion(self, definition: WorkflowDefinition, state: WorkflowState, context: dict = None, max_steps: int = 100) -> WorkflowState
    def pause(self, state: WorkflowState) -> WorkflowState
    def resume(self, state: WorkflowState) -> WorkflowState
    async def provide_human_response(self, definition: WorkflowDefinition, state: WorkflowState, node_id: str, response: Any, context: dict = None) -> WorkflowState
```

### extensions.py
```python
# Type aliases
BeforeNodeHook = Callable[[NodeDefinition, WorkflowState], Awaitable[WorkflowState | None]]
AfterNodeHook = Callable[[NodeDefinition, WorkflowState, dict], Awaitable[dict | None]]
ErrorHook = Callable[[NodeDefinition, WorkflowState, Exception], Awaitable[bool]]
ConditionEvaluator = Callable[[str, dict], bool]
StateTransformer = Callable[[WorkflowState], WorkflowState]

class WorkflowHooks:
    def before_node(self, node_type: str = None) -> Callable
    def after_node(self, node_type: str = None) -> Callable
    def on_error(self, node_type: str = None) -> Callable
    def before_workflow_start(self, func: Callable) -> Callable
    def after_workflow_complete(self, func: Callable) -> Callable
    async def run_before_node(self, node: NodeDefinition, state: WorkflowState) -> WorkflowState
    async def run_after_node(self, node: NodeDefinition, state: WorkflowState, output: dict) -> dict
    async def run_on_error(self, node: NodeDefinition, state: WorkflowState, error: Exception) -> bool

class ConditionRegistry:
    def register(self, prefix: str) -> Callable
    def evaluate(self, condition: str, context: dict) -> bool

class TransformerRegistry:
    def register(self, name: str) -> Callable
    def get(self, name: str) -> StateTransformer | None
    def apply(self, name: str, state: WorkflowState) -> WorkflowState
    def list(self) -> list[str]

workflow_hooks = WorkflowHooks()
condition_registry = ConditionRegistry()
transformer_registry = TransformerRegistry()
```

---

## RAG Pipeline (`app/rag/`)

### chunker.py
```python
@dataclass
class Chunk:
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict
    def to_document(self, source_id: str = None) -> Document

class TextChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]

class FixedSizeChunker(TextChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200)

class SemanticChunker(TextChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, min_chunk_size: int = 100)

class RecursiveChunker(TextChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200)

def create_chunker(strategy: str = "semantic", **kwargs) -> TextChunker
```

### embeddings.py
```python
@dataclass
class EmbeddingResult:
    embeddings: list[list[float]]
    model: str
    usage: dict | None = None

class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int

    @property
    @abstractmethod
    def model_name(self) -> str

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResult

    async def embed_query(self, query: str) -> list[float]
    async def embed_documents(self, documents: list[str]) -> list[list[float]]

class NemotronEmbedder(EmbeddingProvider):
    def __init__(self, api_key: str = None, model: str = NEMOTRON_1B)
    # dimension = 4096

class OpenAIEmbedder(EmbeddingProvider):
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small")
    # dimension = 1536 (small), 3072 (large)

class NoEmbedder(EmbeddingProvider):
    # dimension = 0, for BM25-only

def create_embedder(provider: str = "nemotron", **kwargs) -> EmbeddingProvider
```

### vector_store.py
```python
@dataclass
class Document:
    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict
    created_at: datetime
    def to_dict(self) -> dict

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int

class VectorStore(ABC):
    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]

    @abstractmethod
    async def search(self, query_embedding: list[float], limit: int = 10, filter: dict = None) -> list[SearchResult]

    @abstractmethod
    async def delete(self, document_ids: list[str]) -> int

    @abstractmethod
    async def get(self, document_id: str) -> Document | None

class PgVectorStore(VectorStore):
    def __init__(self, connection_string: str, table_name: str = "documents", embedding_dimension: int = 4096)
    async def initialize(self) -> None
    async def close(self) -> None

class InMemoryVectorStore(VectorStore):
    # For testing
```

### retriever.py
```python
@dataclass
class RetrievalResult:
    documents: list[SearchResult]
    query: str
    strategy: str
    metadata: dict

class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, limit: int = 10, filter: dict = None) -> RetrievalResult

class VectorRetriever(Retriever):
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingProvider)

class BM25Retriever(Retriever):
    def __init__(self, documents: list[Document] = None, k1: float = 1.5, b: float = 0.75)
    def add_documents(self, documents: list[Document]) -> None

class HybridRetriever(Retriever):
    def __init__(self, vector_retriever: VectorRetriever = None, bm25_retriever: BM25Retriever = None, vector_weight: float = 0.5, rrf_k: int = 60)

def create_retriever(strategy: str = "hybrid", vector_store: VectorStore = None, embedder: EmbeddingProvider = None, documents: list[Document] = None, **kwargs) -> Retriever
```

---

## Providers (`app/providers/`)

### LLM Providers

```python
# base.py
@dataclass
class LLMMessage:
    role: str
    content: str

@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict | None = None

class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[LLMMessage]) -> LLMResponse

    @abstractmethod
    async def stream_chat(self, messages: list[LLMMessage]) -> AsyncIterator[str]

    @staticmethod
    def get_available_models() -> list[str]

# openai_provider.py
class OpenAIProvider(BaseLLMProvider):
    MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "o1-preview", "o1-mini"]

# anthropic_provider.py
class AnthropicProvider(BaseLLMProvider):
    MODELS = ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022", ...]

# ollama_provider.py
class OllamaProvider(BaseLLMProvider):
    DEFAULT_MODELS = ["llama3.2", "llama3.1", "mistral", "codellama", "phi3", "qwen2"]
    async def list_local_models(self) -> list[str]

# openai_compatible_provider.py
class OpenAICompatibleProvider(BaseLLMProvider):
    # For Azure, vLLM, LiteLLM, LocalAI, etc.

# factory.py
class LLMProviderFactory:
    @classmethod
    def create(cls, provider_type: ProviderType, api_key: str, model_name: str, base_url: str = None) -> BaseLLMProvider

    @classmethod
    def get_available_models(cls, provider_type: ProviderType) -> list[str]
```

### DataSource Providers

```python
# base.py
@dataclass
class DataSourceContent:
    content: str
    source: str
    metadata: dict | None = None

class BaseDataSourceProvider(ABC):
    @abstractmethod
    async def fetch_content(self, source: str) -> DataSourceContent

    @abstractmethod
    async def validate_source(self, source: str) -> bool

# file_provider.py
class FileDataSourceProvider(BaseDataSourceProvider):
    SUPPORTED_EXTENSIONS = [".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml", ".py", ".js"]
    async def save_uploaded_file(self, filename: str, content: bytes) -> str

# url_provider.py
class URLDataSourceProvider(BaseDataSourceProvider):
    # HTML parsing with BeautifulSoup

# text_provider.py
class TextDataSourceProvider(BaseDataSourceProvider):
    # Inline text content

# gitlab_provider.py
class GitLabDataSourceProvider(BaseDataSourceProvider):
    async def fetch_file(self, project_id: str, file_path: str, ref: str) -> DataSourceContent
    async def fetch_tree(self, project_id: str, path: str, ref: str) -> DataSourceContent
    async def fetch_issues(self, project_id: str, state: str = "all") -> DataSourceContent
    async def fetch_merge_requests(self, project_id: str, state: str = "all") -> DataSourceContent
    async def fetch_repository(self, project_id: str, ref: str, patterns: list[str] = None) -> DataSourceContent

# factory.py
class DataSourceFactory:
    @classmethod
    def create(cls, source_type: DataSourceType, **kwargs) -> BaseDataSourceProvider
```

---

## Services (`app/services/`)

```python
# agent_service.py
class AgentService:
    async def create(self, data: AgentCreate) -> Agent
    async def get_by_id(self, agent_id: str) -> Agent | None
    async def get_all(self) -> list[Agent]
    async def update(self, agent_id: str, data: AgentUpdate) -> Agent | None
    async def delete(self, agent_id: str) -> bool
    async def chat(self, agent_id: str, request: ChatRequest) -> ChatResponse

# datasource_service.py
class DataSourceService:
    async def create(self, data: DataSourceCreate) -> DataSource
    async def get_by_id(self, datasource_id: str) -> DataSource | None
    async def get_all(self) -> list[DataSource]
    async def get_by_ids(self, ids: list[str]) -> list[DataSource]
    async def update(self, datasource_id: str, data: DataSourceUpdate) -> DataSource | None
    async def delete(self, datasource_id: str) -> bool
    async def refresh_content(self, datasource_id: str) -> DataSource | None

# provider_service.py
class ProviderService:
    async def create(self, data: LLMProviderCreate) -> LLMProvider
    async def get_by_id(self, provider_id: str) -> LLMProvider | None
    async def get_all(self, active_only: bool = False) -> list[LLMProvider]
    async def update(self, provider_id: str, data: LLMProviderUpdate) -> LLMProvider | None
    async def delete(self, provider_id: str) -> bool
```

---

## Tasks (`app/tasks/`)

```python
# datasource_tasks.py
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def refresh_datasource(self, datasource_id: str) -> dict

@shared_task
def refresh_all_datasources() -> dict

@shared_task
def cleanup_expired_datasources() -> int

# llm_tasks.py
@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def process_chat_async(self, agent_id: str, messages: list[dict], user_id: str) -> dict

@shared_task
def batch_embed_documents(documents: list[str], model: str = "text-embedding-3-small") -> list[list[float]]

@shared_task
def summarize_conversation(conversation_id: str, max_tokens: int = 500) -> str

# webhook_tasks.py
@shared_task(bind=True, max_retries=5, default_retry_delay=30, retry_backoff=True)
def deliver_webhook(self, webhook_url: str, event_type: str, payload: dict, headers: dict = None) -> dict

@shared_task
def cleanup_failed_webhooks(hours_old: int = 24) -> int
```
