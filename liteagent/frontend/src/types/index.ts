export type ProviderType = "openai" | "anthropic" | "ollama";
export type DataSourceType = "file" | "url" | "text";

export interface LLMProvider {
  id: string;
  name: string;
  provider_type: ProviderType;
  model_name: string;
  api_key?: string;
  base_url?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface LLMProviderCreate {
  name: string;
  provider_type: ProviderType;
  model_name: string;
  api_key?: string;
  base_url?: string;
}

export interface DataSource {
  id: string;
  name: string;
  source_type: DataSourceType;
  content?: string;
  source_path?: string;
  created_at: string;
  updated_at: string;
}

export interface DataSourceCreate {
  name: string;
  source_type: DataSourceType;
  content?: string;
  source_path?: string;
}

export interface Agent {
  id: string;
  name: string;
  description?: string;
  system_prompt: string;
  provider_id: string;
  provider: LLMProvider;
  datasources: DataSource[];
  created_at: string;
  updated_at: string;
}

export interface AgentCreate {
  name: string;
  description?: string;
  system_prompt: string;
  provider_id: string;
  datasource_ids: string[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  message: string;
  conversation_history: ChatMessage[];
}

export interface ChatResponse {
  response: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ProviderTypeOption {
  value: ProviderType;
  label: string;
}

export interface DataSourceTypeOption {
  value: DataSourceType;
  label: string;
}
