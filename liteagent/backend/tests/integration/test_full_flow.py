"""
Integration tests for the full application flow.
Tests end-to-end scenarios with mocked external services.
"""
import pytest
import respx
import base64
from httpx import AsyncClient, Response


class TestFullAgentFlow:
    """Integration tests for creating and using an agent."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_create_provider_agent_and_chat(self, client: AsyncClient):
        """Test the full flow: create provider, agent, and chat."""
        # 1. Create a provider
        provider_data = {
            "name": "Test OpenAI Provider",
            "provider_type": "openai",
            "model_name": "gpt-4o",
            "api_key": "sk-test-key",
        }
        provider_response = await client.post("/api/providers/", json=provider_data)
        assert provider_response.status_code == 200
        provider_id = provider_response.json()["id"]

        # 2. Create an agent
        agent_data = {
            "name": "Test Agent",
            "description": "A test assistant",
            "system_prompt": "You are a helpful test assistant.",
            "provider_id": provider_id,
        }
        agent_response = await client.post("/api/agents/", json=agent_data)
        assert agent_response.status_code == 200
        agent_id = agent_response.json()["id"]

        # 3. Mock OpenAI API for chat
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! How can I help you today?",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                },
            )
        )

        # 4. Chat with the agent
        chat_data = {
            "message": "Hello!",
            "conversation_history": [],
        }
        chat_response = await client.post(f"/api/agents/{agent_id}/chat", json=chat_data)
        assert chat_response.status_code == 200
        assert "response" in chat_response.json()

        # 5. Cleanup - delete agent and provider
        await client.delete(f"/api/agents/{agent_id}")
        await client.delete(f"/api/providers/{provider_id}")

    @pytest.mark.asyncio
    @respx.mock
    async def test_agent_with_datasource(self, client: AsyncClient):
        """Test creating an agent with a data source."""
        # 1. Create a text datasource
        datasource_data = {
            "name": "Product Documentation",
            "source_type": "text",
            "content": "Our product is an AI-powered assistant that helps with coding.",
        }
        ds_response = await client.post("/api/datasources/", json=datasource_data)
        assert ds_response.status_code == 200
        datasource_id = ds_response.json()["id"]

        # 2. Create a provider
        provider_data = {
            "name": "OpenAI",
            "provider_type": "openai",
            "model_name": "gpt-4o",
            "api_key": "sk-test",
        }
        prov_response = await client.post("/api/providers/", json=provider_data)
        provider_id = prov_response.json()["id"]

        # 3. Create agent with datasource
        agent_data = {
            "name": "Product Expert",
            "system_prompt": "You are an expert on our product. Use the provided context.",
            "provider_id": provider_id,
            "datasource_ids": [datasource_id],
        }
        agent_response = await client.post("/api/agents/", json=agent_data)
        assert agent_response.status_code == 200
        agent = agent_response.json()
        assert len(agent["datasources"]) == 1
        assert agent["datasources"][0]["name"] == "Product Documentation"


class TestGitLabIntegration:
    """Integration tests for GitLab datasource."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_create_and_refresh_gitlab_datasource(self, client: AsyncClient):
        """Test creating and refreshing a GitLab datasource."""
        # Mock GitLab API - use numeric project ID
        file_content = base64.b64encode(b"# Project Documentation\n\nThis is the README.").decode()
        respx.get("https://gitlab.com/api/v4/projects/12345/repository/files/README.md").mock(
            return_value=Response(
                200,
                json={
                    "file_name": "README.md",
                    "file_path": "README.md",
                    "content": file_content,
                    "encoding": "base64",
                },
            )
        )

        # Create GitLab datasource with numeric project ID
        datasource_data = {
            "name": "My GitLab Repo",
            "source_type": "gitlab",
            "source_path": "project:12345/file:README.md@main",
            "gitlab_url": "https://gitlab.com",
            "gitlab_token": "glpat-xxxxxxxxxxxx",
        }
        response = await client.post("/api/datasources/", json=datasource_data)
        assert response.status_code == 200

        datasource = response.json()
        assert datasource["source_type"] == "gitlab"
        assert "README" in datasource["content"]

        # Refresh the datasource
        refresh_response = await client.post(f"/api/datasources/{datasource['id']}/refresh")
        assert refresh_response.status_code == 200


class TestOpenAICompatibleFlow:
    """Integration tests for OpenAI-compatible providers."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_azure_openai_chat(self, client: AsyncClient):
        """Test chatting with Azure OpenAI."""
        # 1. Create Azure OpenAI provider
        provider_data = {
            "name": "Azure GPT-4",
            "provider_type": "openai_compatible",
            "model_name": "gpt-4-deployment",
            "api_key": "azure-api-key",
            "base_url": "https://my-resource.openai.azure.com/openai/deployments/gpt-4/",
        }
        prov_response = await client.post("/api/providers/", json=provider_data)
        assert prov_response.status_code == 200
        provider_id = prov_response.json()["id"]

        # 2. Create agent
        agent_data = {
            "name": "Azure Agent",
            "system_prompt": "You are an Azure-powered assistant.",
            "provider_id": provider_id,
        }
        agent_response = await client.post("/api/agents/", json=agent_data)
        agent_id = agent_response.json()["id"]

        # 3. Mock Azure OpenAI API
        respx.post("https://my-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "id": "chatcmpl-azure",
                    "object": "chat.completion",
                    "model": "gpt-4-deployment",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Hello from Azure!",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
        )

        # 4. Chat with agent
        chat_response = await client.post(
            f"/api/agents/{agent_id}/chat",
            json={"message": "Hello!", "conversation_history": []},
        )
        assert chat_response.status_code == 200


class TestMetaEndpoints:
    """Integration tests for meta information endpoints."""

    @pytest.mark.asyncio
    async def test_get_all_provider_types(self, client: AsyncClient):
        """Test getting all supported provider types."""
        response = await client.get("/api/meta/provider-types")
        assert response.status_code == 200

        data = response.json()
        types = [t["value"] for t in data["provider_types"]]

        assert "openai" in types
        assert "anthropic" in types
        assert "ollama" in types
        assert "openai_compatible" in types

    @pytest.mark.asyncio
    async def test_get_all_datasource_types(self, client: AsyncClient):
        """Test getting all supported datasource types."""
        response = await client.get("/api/meta/datasource-types")
        assert response.status_code == 200

        data = response.json()
        types = [t["value"] for t in data["datasource_types"]]

        assert "text" in types
        assert "url" in types
        assert "file" in types
        assert "gitlab" in types
