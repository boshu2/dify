"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2, Edit2 } from "lucide-react";
import { Header } from "@/components/header";
import { Button, Input, Select, Card, Modal } from "@/components/ui";
import { providersAPI, metaAPI } from "@/lib/api";
import type { LLMProvider, LLMProviderCreate, ProviderType } from "@/types";

export default function ProvidersPage() {
  const queryClient = useQueryClient();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingProvider, setEditingProvider] = useState<LLMProvider | null>(
    null
  );

  // Queries
  const { data: providers = [], isLoading } = useQuery({
    queryKey: ["providers"],
    queryFn: providersAPI.list,
  });

  const { data: providerTypesData } = useQuery({
    queryKey: ["providerTypes"],
    queryFn: metaAPI.getProviderTypes,
  });

  // Mutations
  const createMutation = useMutation({
    mutationFn: providersAPI.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["providers"] });
      setIsModalOpen(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: providersAPI.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["providers"] });
    },
  });

  const providerTypes =
    providerTypesData?.provider_types.map((pt) => ({
      value: pt.value,
      label: pt.label,
    })) || [];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              LLM Providers
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Manage your LLM provider connections
            </p>
          </div>
          <Button onClick={() => setIsModalOpen(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Add Provider
          </Button>
        </div>

        {isLoading ? (
          <div className="text-center py-12">Loading...</div>
        ) : providers.length === 0 ? (
          <Card className="text-center py-12">
            <p className="text-gray-500 dark:text-gray-400 mb-4">
              No providers configured yet
            </p>
            <Button onClick={() => setIsModalOpen(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Add your first provider
            </Button>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {providers.map((provider) => (
              <Card key={provider.id}>
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white">
                      {provider.name}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {provider.provider_type} / {provider.model_name}
                    </p>
                    <span
                      className={`inline-block mt-2 px-2 py-1 text-xs rounded-full ${
                        provider.is_active
                          ? "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300"
                          : "bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300"
                      }`}
                    >
                      {provider.is_active ? "Active" : "Inactive"}
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteMutation.mutate(provider.id)}
                  >
                    <Trash2 className="w-4 h-4 text-red-500" />
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* Add Provider Modal */}
        <ProviderModal
          isOpen={isModalOpen}
          onClose={() => {
            setIsModalOpen(false);
            setEditingProvider(null);
          }}
          onSubmit={(data) => createMutation.mutate(data)}
          providerTypes={providerTypes}
          isLoading={createMutation.isPending}
        />
      </main>
    </div>
  );
}

interface ProviderModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: LLMProviderCreate) => void;
  providerTypes: { value: string; label: string }[];
  isLoading: boolean;
}

function ProviderModal({
  isOpen,
  onClose,
  onSubmit,
  providerTypes,
  isLoading,
}: ProviderModalProps) {
  const [formData, setFormData] = useState<LLMProviderCreate>({
    name: "",
    provider_type: "openai",
    model_name: "",
    api_key: "",
    base_url: "",
  });

  const [selectedProviderType, setSelectedProviderType] =
    useState<ProviderType>("openai");

  // Fetch models for selected provider
  const { data: modelsData } = useQuery({
    queryKey: ["models", selectedProviderType],
    queryFn: () => metaAPI.getModelsForProvider(selectedProviderType),
    enabled: !!selectedProviderType,
  });

  const models =
    modelsData?.models.map((m) => ({ value: m, label: m })) || [];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Add LLM Provider">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Name"
          placeholder="My OpenAI Provider"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          required
        />

        <Select
          label="Provider Type"
          options={providerTypes}
          value={formData.provider_type}
          onChange={(e) => {
            const type = e.target.value as ProviderType;
            setSelectedProviderType(type);
            setFormData({
              ...formData,
              provider_type: type,
              model_name: "",
            });
          }}
        />

        <Select
          label="Model"
          options={models}
          value={formData.model_name}
          onChange={(e) =>
            setFormData({ ...formData, model_name: e.target.value })
          }
          placeholder="Select a model"
        />

        {selectedProviderType !== "ollama" && (
          <Input
            label="API Key"
            type="password"
            placeholder="sk-..."
            value={formData.api_key}
            onChange={(e) =>
              setFormData({ ...formData, api_key: e.target.value })
            }
          />
        )}

        {selectedProviderType === "ollama" && (
          <Input
            label="Base URL"
            placeholder="http://localhost:11434"
            value={formData.base_url}
            onChange={(e) =>
              setFormData({ ...formData, base_url: e.target.value })
            }
          />
        )}

        <div className="flex justify-end gap-2 pt-4">
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" loading={isLoading}>
            Add Provider
          </Button>
        </div>
      </form>
    </Modal>
  );
}
