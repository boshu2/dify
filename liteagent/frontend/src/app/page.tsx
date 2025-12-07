"use client";

import Link from "next/link";
import { Bot, Database, Cpu } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            LiteAgent
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Lightweight LLM Agent Platform
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Providers Card */}
          <Link
            href="/providers"
            className="block p-6 bg-white dark:bg-gray-800 rounded-lg shadow hover:shadow-lg transition-shadow"
          >
            <div className="flex items-center gap-4">
              <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg">
                <Cpu className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  LLM Providers
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  Connect OpenAI, Anthropic, Ollama
                </p>
              </div>
            </div>
          </Link>

          {/* Data Sources Card */}
          <Link
            href="/datasources"
            className="block p-6 bg-white dark:bg-gray-800 rounded-lg shadow hover:shadow-lg transition-shadow"
          >
            <div className="flex items-center gap-4">
              <div className="p-3 bg-green-100 dark:bg-green-900 rounded-lg">
                <Database className="w-8 h-8 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Data Sources
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  Add files, URLs, or text
                </p>
              </div>
            </div>
          </Link>

          {/* Agents Card */}
          <Link
            href="/agents"
            className="block p-6 bg-white dark:bg-gray-800 rounded-lg shadow hover:shadow-lg transition-shadow"
          >
            <div className="flex items-center gap-4">
              <div className="p-3 bg-purple-100 dark:bg-purple-900 rounded-lg">
                <Bot className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Agents
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  Create and chat with agents
                </p>
              </div>
            </div>
          </Link>
        </div>

        {/* Quick Start */}
        <div className="mt-12 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Quick Start
          </h2>
          <ol className="list-decimal list-inside space-y-3 text-gray-700 dark:text-gray-300">
            <li>
              <strong>Add an LLM Provider</strong> - Connect your OpenAI,
              Anthropic, or Ollama API
            </li>
            <li>
              <strong>Add Data Sources</strong> - Upload files, add URLs, or
              paste text content
            </li>
            <li>
              <strong>Create an Agent</strong> - Pick a provider, attach data
              sources, set a system prompt
            </li>
            <li>
              <strong>Chat!</strong> - Test your agent with the built-in chat
              interface
            </li>
          </ol>
        </div>
      </main>
    </div>
  );
}
