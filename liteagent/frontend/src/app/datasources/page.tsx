"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2, RefreshCw, FileText, Link2, Type } from "lucide-react";
import { Header } from "@/components/header";
import {
  Button,
  Input,
  Select,
  Textarea,
  Card,
  Modal,
} from "@/components/ui";
import { datasourcesAPI, metaAPI } from "@/lib/api";
import type { DataSource, DataSourceCreate, DataSourceType } from "@/types";

const sourceTypeIcons: Record<DataSourceType, typeof FileText> = {
  file: FileText,
  url: Link2,
  text: Type,
};

export default function DataSourcesPage() {
  const queryClient = useQueryClient();
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Queries
  const { data: datasources = [], isLoading } = useQuery({
    queryKey: ["datasources"],
    queryFn: datasourcesAPI.list,
  });

  const { data: sourceTypesData } = useQuery({
    queryKey: ["datasourceTypes"],
    queryFn: metaAPI.getDataSourceTypes,
  });

  // Mutations
  const createMutation = useMutation({
    mutationFn: datasourcesAPI.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasources"] });
      setIsModalOpen(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: datasourcesAPI.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasources"] });
    },
  });

  const refreshMutation = useMutation({
    mutationFn: datasourcesAPI.refresh,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasources"] });
    },
  });

  const sourceTypes =
    sourceTypesData?.datasource_types.map((st) => ({
      value: st.value,
      label: st.label,
    })) || [];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Data Sources
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Add files, URLs, or text content
            </p>
          </div>
          <Button onClick={() => setIsModalOpen(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Add Data Source
          </Button>
        </div>

        {isLoading ? (
          <div className="text-center py-12">Loading...</div>
        ) : datasources.length === 0 ? (
          <Card className="text-center py-12">
            <p className="text-gray-500 dark:text-gray-400 mb-4">
              No data sources added yet
            </p>
            <Button onClick={() => setIsModalOpen(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Add your first data source
            </Button>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasources.map((ds) => {
              const Icon = sourceTypeIcons[ds.source_type];
              return (
                <Card key={ds.id}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
                        <Icon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white">
                          {ds.name}
                        </h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          {ds.source_type}
                        </p>
                        {ds.source_path && (
                          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1 truncate max-w-[200px]">
                            {ds.source_path}
                          </p>
                        )}
                        {ds.content && (
                          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                            {ds.content.length} characters
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="flex gap-1">
                      {ds.source_type === "url" && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => refreshMutation.mutate(ds.id)}
                          disabled={refreshMutation.isPending}
                        >
                          <RefreshCw
                            className={`w-4 h-4 text-blue-500 ${
                              refreshMutation.isPending ? "animate-spin" : ""
                            }`}
                          />
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => deleteMutation.mutate(ds.id)}
                      >
                        <Trash2 className="w-4 h-4 text-red-500" />
                      </Button>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        {/* Add Data Source Modal */}
        <DataSourceModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onSubmit={(data) => createMutation.mutate(data)}
          sourceTypes={sourceTypes}
          isLoading={createMutation.isPending}
        />
      </main>
    </div>
  );
}

interface DataSourceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: DataSourceCreate) => void;
  sourceTypes: { value: string; label: string }[];
  isLoading: boolean;
}

function DataSourceModal({
  isOpen,
  onClose,
  onSubmit,
  sourceTypes,
  isLoading,
}: DataSourceModalProps) {
  const [formData, setFormData] = useState<DataSourceCreate>({
    name: "",
    source_type: "text",
    content: "",
    source_path: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Add Data Source">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Name"
          placeholder="My Knowledge Base"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          required
        />

        <Select
          label="Source Type"
          options={sourceTypes}
          value={formData.source_type}
          onChange={(e) =>
            setFormData({
              ...formData,
              source_type: e.target.value as DataSourceType,
              content: "",
              source_path: "",
            })
          }
        />

        {formData.source_type === "url" && (
          <Input
            label="URL"
            type="url"
            placeholder="https://example.com/page"
            value={formData.source_path}
            onChange={(e) =>
              setFormData({ ...formData, source_path: e.target.value })
            }
            required
          />
        )}

        {formData.source_type === "text" && (
          <Textarea
            label="Content"
            placeholder="Paste your text content here..."
            value={formData.content}
            onChange={(e) =>
              setFormData({ ...formData, content: e.target.value })
            }
            rows={8}
            required
          />
        )}

        {formData.source_type === "file" && (
          <div className="text-sm text-gray-500 dark:text-gray-400 p-4 bg-gray-100 dark:bg-gray-700 rounded-lg">
            File upload coming soon. Use URL or Text for now.
          </div>
        )}

        <div className="flex justify-end gap-2 pt-4">
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="submit"
            loading={isLoading}
            disabled={formData.source_type === "file"}
          >
            Add Data Source
          </Button>
        </div>
      </form>
    </Modal>
  );
}
