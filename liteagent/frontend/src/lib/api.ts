import type {
  LLMProvider,
  LLMProviderCreate,
  DataSource,
  DataSourceCreate,
  Agent,
  AgentCreate,
  ChatRequest,
  ChatResponse,
  ProviderType,
  ProviderTypeOption,
  DataSourceTypeOption,
} from "@/types";

const API_BASE = "/api";

async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

// Providers API
export const providersAPI = {
  list: () => fetchAPI<LLMProvider[]>("/providers/"),
  get: (id: string) => fetchAPI<LLMProvider>(`/providers/${id}`),
  create: (data: LLMProviderCreate) =>
    fetchAPI<LLMProvider>("/providers/", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  update: (id: string, data: Partial<LLMProviderCreate>) =>
    fetchAPI<LLMProvider>(`/providers/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    fetchAPI<{ status: string }>(`/providers/${id}`, { method: "DELETE" }),
};

// DataSources API
export const datasourcesAPI = {
  list: () => fetchAPI<DataSource[]>("/datasources/"),
  get: (id: string) => fetchAPI<DataSource>(`/datasources/${id}`),
  create: (data: DataSourceCreate) =>
    fetchAPI<DataSource>("/datasources/", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  update: (id: string, data: Partial<DataSourceCreate>) =>
    fetchAPI<DataSource>(`/datasources/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    fetchAPI<{ status: string }>(`/datasources/${id}`, { method: "DELETE" }),
  refresh: (id: string) =>
    fetchAPI<DataSource>(`/datasources/${id}/refresh`, { method: "POST" }),
};

// Agents API
export const agentsAPI = {
  list: () => fetchAPI<Agent[]>("/agents/"),
  get: (id: string) => fetchAPI<Agent>(`/agents/${id}`),
  create: (data: AgentCreate) =>
    fetchAPI<Agent>("/agents/", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  update: (id: string, data: Partial<AgentCreate>) =>
    fetchAPI<Agent>(`/agents/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    fetchAPI<{ status: string }>(`/agents/${id}`, { method: "DELETE" }),
  chat: (id: string, data: ChatRequest) =>
    fetchAPI<ChatResponse>(`/agents/${id}/chat`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
};

// Meta API
export const metaAPI = {
  getProviderTypes: () =>
    fetchAPI<{ provider_types: ProviderTypeOption[] }>("/meta/provider-types"),
  getModelsForProvider: (providerType: ProviderType) =>
    fetchAPI<{ models: string[] }>(
      `/meta/provider-types/${providerType}/models`
    ),
  getDataSourceTypes: () =>
    fetchAPI<{ datasource_types: DataSourceTypeOption[] }>(
      "/meta/datasource-types"
    ),
};
