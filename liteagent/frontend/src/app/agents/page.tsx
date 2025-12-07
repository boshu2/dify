"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2, MessageSquare, Bot } from "lucide-react";
import { Header } from "@/components/header";
import {
  Button,
  Input,
  Select,
  Textarea,
  Card,
  Modal,
} from "@/components/ui";
import { agentsAPI, providersAPI, datasourcesAPI } from "@/lib/api";
import type { Agent, AgentCreate, ChatMessage } from "@/types";

export default function AgentsPage() {
  const queryClient = useQueryClient();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [chatAgent, setChatAgent] = useState<Agent | null>(null);

  // Queries
  const { data: agents = [], isLoading } = useQuery({
    queryKey: ["agents"],
    queryFn: agentsAPI.list,
  });

  const { data: providers = [] } = useQuery({
    queryKey: ["providers"],
    queryFn: providersAPI.list,
  });

  const { data: datasources = [] } = useQuery({
    queryKey: ["datasources"],
    queryFn: datasourcesAPI.list,
  });

  // Mutations
  const createMutation = useMutation({
    mutationFn: agentsAPI.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      setIsModalOpen(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: agentsAPI.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agents"] });
    },
  });

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Agents
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Create and chat with your AI agents
            </p>
          </div>
          <Button onClick={() => setIsModalOpen(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Create Agent
          </Button>
        </div>

        {providers.length === 0 && (
          <Card className="mb-6 bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800">
            <p className="text-yellow-800 dark:text-yellow-200">
              You need to add an LLM provider before creating agents.
            </p>
          </Card>
        )}

        {isLoading ? (
          <div className="text-center py-12">Loading...</div>
        ) : agents.length === 0 ? (
          <Card className="text-center py-12">
            <p className="text-gray-500 dark:text-gray-400 mb-4">
              No agents created yet
            </p>
            <Button
              onClick={() => setIsModalOpen(true)}
              disabled={providers.length === 0}
            >
              <Plus className="w-4 h-4 mr-2" />
              Create your first agent
            </Button>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {agents.map((agent) => (
              <Card key={agent.id}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-purple-100 dark:bg-purple-900 rounded-lg">
                      <Bot className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">
                        {agent.name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {agent.provider.provider_type} / {agent.provider.model_name}
                      </p>
                      {agent.description && (
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                          {agent.description}
                        </p>
                      )}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteMutation.mutate(agent.id)}
                  >
                    <Trash2 className="w-4 h-4 text-red-500" />
                  </Button>
                </div>

                {agent.datasources.length > 0 && (
                  <div className="mb-4">
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                      Data Sources:
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {agent.datasources.map((ds) => (
                        <span
                          key={ds.id}
                          className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 rounded"
                        >
                          {ds.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <Button
                  className="w-full"
                  variant="secondary"
                  onClick={() => setChatAgent(agent)}
                >
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Chat
                </Button>
              </Card>
            ))}
          </div>
        )}

        {/* Create Agent Modal */}
        <AgentModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onSubmit={(data) => createMutation.mutate(data)}
          providers={providers}
          datasources={datasources}
          isLoading={createMutation.isPending}
        />

        {/* Chat Modal */}
        {chatAgent && (
          <ChatModal
            agent={chatAgent}
            onClose={() => setChatAgent(null)}
          />
        )}
      </main>
    </div>
  );
}

interface AgentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: AgentCreate) => void;
  providers: { id: string; name: string; provider_type: string; model_name: string }[];
  datasources: { id: string; name: string }[];
  isLoading: boolean;
}

function AgentModal({
  isOpen,
  onClose,
  onSubmit,
  providers,
  datasources,
  isLoading,
}: AgentModalProps) {
  const [formData, setFormData] = useState<AgentCreate>({
    name: "",
    description: "",
    system_prompt: "You are a helpful AI assistant.",
    provider_id: "",
    datasource_ids: [],
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const toggleDatasource = (id: string) => {
    setFormData((prev) => ({
      ...prev,
      datasource_ids: prev.datasource_ids.includes(id)
        ? prev.datasource_ids.filter((dsId) => dsId !== id)
        : [...prev.datasource_ids, id],
    }));
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Create Agent" className="max-w-xl">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Name"
          placeholder="My AI Assistant"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          required
        />

        <Input
          label="Description (optional)"
          placeholder="A helpful assistant that..."
          value={formData.description}
          onChange={(e) =>
            setFormData({ ...formData, description: e.target.value })
          }
        />

        <Select
          label="LLM Provider"
          options={providers.map((p) => ({
            value: p.id,
            label: `${p.name} (${p.provider_type}/${p.model_name})`,
          }))}
          value={formData.provider_id}
          onChange={(e) =>
            setFormData({ ...formData, provider_id: e.target.value })
          }
          placeholder="Select a provider"
          required
        />

        <Textarea
          label="System Prompt"
          placeholder="You are a helpful AI assistant..."
          value={formData.system_prompt}
          onChange={(e) =>
            setFormData({ ...formData, system_prompt: e.target.value })
          }
          rows={4}
          required
        />

        {datasources.length > 0 && (
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Data Sources (optional)
            </label>
            <div className="flex flex-wrap gap-2">
              {datasources.map((ds) => (
                <button
                  key={ds.id}
                  type="button"
                  onClick={() => toggleDatasource(ds.id)}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    formData.datasource_ids.includes(ds.id)
                      ? "bg-blue-100 border-blue-500 text-blue-700 dark:bg-blue-900 dark:border-blue-400 dark:text-blue-300"
                      : "bg-gray-100 border-gray-300 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300"
                  }`}
                >
                  {ds.name}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-4">
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" loading={isLoading}>
            Create Agent
          </Button>
        </div>
      </form>
    </Modal>
  );
}

interface ChatModalProps {
  agent: Agent;
  onClose: () => void;
}

function ChatModal({ agent, onClose }: ChatModalProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");

  const chatMutation = useMutation({
    mutationFn: (message: string) =>
      agentsAPI.chat(agent.id, {
        message,
        conversation_history: messages,
      }),
    onSuccess: (response) => {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: response.response },
      ]);
    },
  });

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || chatMutation.isPending) return;

    const userMessage: ChatMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    chatMutation.mutate(input);
  };

  return (
    <Modal
      isOpen={true}
      onClose={onClose}
      title={`Chat with ${agent.name}`}
      className="max-w-2xl h-[80vh] flex flex-col"
    >
      {/* Messages */}
      <div className="flex-1 overflow-auto space-y-4 mb-4 min-h-[300px]">
        {messages.length === 0 && (
          <p className="text-center text-gray-500 dark:text-gray-400 py-8">
            Start a conversation with {agent.name}
          </p>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}
        {chatMutation.isPending && (
          <div className="flex justify-start">
            <div className="bg-gray-100 dark:bg-gray-700 rounded-lg px-4 py-2">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.1s]" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]" />
              </div>
            </div>
          </div>
        )}
        {chatMutation.isError && (
          <div className="text-center text-red-500 py-2">
            Error: {(chatMutation.error as Error).message}
          </div>
        )}
      </div>

      {/* Input */}
      <form onSubmit={handleSend} className="flex gap-2">
        <Input
          className="flex-1"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={chatMutation.isPending}
        />
        <Button type="submit" loading={chatMutation.isPending}>
          Send
        </Button>
      </form>
    </Modal>
  );
}
