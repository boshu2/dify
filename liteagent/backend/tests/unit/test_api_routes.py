"""
Unit tests for API routes.
Tests the REST API endpoints.
"""
import base64
import pytest
import respx
from httpx import AsyncClient, Response


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """Test health check returns OK."""
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.mark.asyncio
    async def test_root(self, client: AsyncClient):
        """Test root endpoint returns app info."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestProvidersAPI:
    """Tests for providers API endpoints."""

    @pytest.mark.asyncio
    async def test_list_providers_empty(self, client: AsyncClient):
        """Test listing providers when none exist."""
        response = await client.get("/api/providers/")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_create_provider(self, client: AsyncClient, provider_data):
        """Test creating a new provider."""
        data = provider_data()
        response = await client.post("/api/providers/", json=data)

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == data["name"]
        assert result["provider_type"] == data["provider_type"]
        assert result["model_name"] == data["model_name"]
        # API key should be masked in response
        assert result["api_key"] == "***hidden***"
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_openai_compatible_provider(self, client: AsyncClient):
        """Test creating an OpenAI-compatible provider."""
        data = {
            "name": "Azure OpenAI",
            "provider_type": "openai_compatible",
            "model_name": "gpt-4-deployment",
            "api_key": "azure-key",
            "base_url": "https://my-resource.openai.azure.com/openai/deployments/gpt-4/",
        }
        response = await client.post("/api/providers/", json=data)

        assert response.status_code == 200
        result = response.json()
        assert result["provider_type"] == "openai_compatible"
        assert result["base_url"] == data["base_url"]

    @pytest.mark.asyncio
    async def test_get_provider(self, client: AsyncClient, provider_data):
        """Test getting a specific provider."""
        # First create
        data = provider_data()
        create_response = await client.post("/api/providers/", json=data)
        provider_id = create_response.json()["id"]

        # Then get
        response = await client.get(f"/api/providers/{provider_id}")
        assert response.status_code == 200
        assert response.json()["id"] == provider_id

    @pytest.mark.asyncio
    async def test_get_provider_not_found(self, client: AsyncClient):
        """Test getting non-existent provider returns 404."""
        response = await client.get("/api/providers/non-existent-id")
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "error_code" in detail
        assert detail["error_code"] == "PROVIDER_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_update_provider(self, client: AsyncClient, provider_data):
        """Test updating a provider."""
        # Create
        data = provider_data()
        create_response = await client.post("/api/providers/", json=data)
        provider_id = create_response.json()["id"]

        # Update
        update_data = {"name": "Updated Provider Name"}
        response = await client.patch(f"/api/providers/{provider_id}", json=update_data)

        assert response.status_code == 200
        assert response.json()["name"] == "Updated Provider Name"

    @pytest.mark.asyncio
    async def test_delete_provider(self, client: AsyncClient, provider_data):
        """Test deleting a provider."""
        # Create
        data = provider_data()
        create_response = await client.post("/api/providers/", json=data)
        provider_id = create_response.json()["id"]

        # Delete
        response = await client.delete(f"/api/providers/{provider_id}")
        assert response.status_code == 200

        # Verify deleted
        get_response = await client.get(f"/api/providers/{provider_id}")
        assert get_response.status_code == 404


class TestDataSourcesAPI:
    """Tests for datasources API endpoints."""

    @pytest.mark.asyncio
    async def test_list_datasources_empty(self, client: AsyncClient):
        """Test listing datasources when none exist."""
        response = await client.get("/api/datasources/")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_create_text_datasource(self, client: AsyncClient, datasource_data):
        """Test creating a text datasource."""
        data = datasource_data(source_type="text", content="Test content")
        response = await client.post("/api/datasources/", json=data)

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == data["name"]
        assert result["source_type"] == "text"
        assert result["content"] == "Test content"

    @pytest.mark.asyncio
    @respx.mock
    async def test_create_gitlab_datasource(self, client: AsyncClient):
        """Test creating a GitLab datasource."""
        # Mock GitLab API response
        file_content = base64.b64encode(b"# My Project\n\nThis is a README.").decode()
        respx.get("https://gitlab.com/api/v4/projects/123/repository/files/README.md").mock(
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

        data = {
            "name": "My GitLab Repo",
            "source_type": "gitlab",
            "source_path": "project:123/file:README.md@main",
            "gitlab_url": "https://gitlab.com",
            "gitlab_token": "glpat-xxxxxxxxxxxx",
        }
        response = await client.post("/api/datasources/", json=data)

        assert response.status_code == 200
        result = response.json()
        assert result["source_type"] == "gitlab"

    @pytest.mark.asyncio
    async def test_refresh_url_datasource(self, client: AsyncClient, datasource_data):
        """Test refreshing a URL datasource."""
        # This would need mocking of the URL fetch
        pass  # Implemented with mocks in integration tests


class TestAgentsAPI:
    """Tests for agents API endpoints."""

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, client: AsyncClient):
        """Test listing agents when none exist."""
        response = await client.get("/api/agents/")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_create_agent(self, client: AsyncClient, provider_data, agent_data):
        """Test creating an agent."""
        # First create a provider
        prov_data = provider_data()
        prov_response = await client.post("/api/providers/", json=prov_data)
        provider_id = prov_response.json()["id"]

        # Create agent
        data = agent_data(provider_id=provider_id)
        response = await client.post("/api/agents/", json=data)

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == data["name"]
        assert result["provider_id"] == provider_id

    @pytest.mark.asyncio
    async def test_create_agent_without_provider_fails(self, client: AsyncClient, agent_data):
        """Test creating agent without valid provider fails."""
        data = agent_data(provider_id="non-existent-provider")
        response = await client.post("/api/agents/", json=data)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_with_agent(self, client: AsyncClient, provider_data, agent_data):
        """Test chatting with an agent."""
        # This requires mocking the LLM provider
        pass  # Implemented with mocks in integration tests


class TestMetaAPI:
    """Tests for meta API endpoints."""

    @pytest.mark.asyncio
    async def test_get_provider_types(self, client: AsyncClient):
        """Test getting all provider types."""
        response = await client.get("/api/meta/provider-types")
        assert response.status_code == 200

        data = response.json()
        assert "provider_types" in data

        types = [pt["value"] for pt in data["provider_types"]]
        assert "openai" in types
        assert "anthropic" in types
        assert "ollama" in types
        assert "openai_compatible" in types

    @pytest.mark.asyncio
    async def test_get_models_for_openai(self, client: AsyncClient):
        """Test getting models for OpenAI provider."""
        response = await client.get("/api/meta/provider-types/openai/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "gpt-4o" in data["models"]

    @pytest.mark.asyncio
    async def test_get_models_for_openai_compatible(self, client: AsyncClient):
        """Test getting models for OpenAI-compatible provider returns empty."""
        response = await client.get("/api/meta/provider-types/openai_compatible/models")
        assert response.status_code == 200

        data = response.json()
        assert data["models"] == []  # User provides their own model names

    @pytest.mark.asyncio
    async def test_get_datasource_types(self, client: AsyncClient):
        """Test getting all datasource types."""
        response = await client.get("/api/meta/datasource-types")
        assert response.status_code == 200

        data = response.json()
        assert "datasource_types" in data

        types = [dt["value"] for dt in data["datasource_types"]]
        assert "text" in types
        assert "url" in types
        assert "file" in types
        assert "gitlab" in types
