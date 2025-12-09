"""
Unit tests for DataSource providers.
Tests the datasource abstraction layer including GitLab connector.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.providers.datasource.base import BaseDataSourceProvider, DataSourceContent
from app.providers.datasource.factory import DataSourceFactory
from app.providers.datasource.text_provider import TextDataSourceProvider
from app.providers.datasource.url_provider import URLDataSourceProvider
from app.providers.datasource.file_provider import FileDataSourceProvider
from app.schemas.datasource import DataSourceType


class TestDataSourceContent:
    """Tests for DataSourceContent dataclass."""

    def test_create_content(self):
        """Test creating datasource content."""
        content = DataSourceContent(
            content="Test content",
            source="test_source",
            metadata={"key": "value"},
        )
        assert content.content == "Test content"
        assert content.source == "test_source"
        assert content.metadata["key"] == "value"

    def test_content_without_metadata(self):
        """Test creating content without metadata."""
        content = DataSourceContent(content="Test", source="src")
        assert content.metadata is None


class TestDataSourceFactory:
    """Tests for DataSourceFactory."""

    def test_create_text_provider(self):
        """Test creating a text provider."""
        provider = DataSourceFactory.create(DataSourceType.TEXT)
        assert isinstance(provider, TextDataSourceProvider)

    def test_create_url_provider(self):
        """Test creating a URL provider."""
        provider = DataSourceFactory.create(DataSourceType.URL)
        assert isinstance(provider, URLDataSourceProvider)

    def test_create_file_provider(self):
        """Test creating a file provider."""
        provider = DataSourceFactory.create(DataSourceType.FILE)
        assert isinstance(provider, FileDataSourceProvider)

    def test_create_gitlab_provider(self):
        """Test creating a GitLab provider."""
        provider = DataSourceFactory.create(
            DataSourceType.GITLAB,
            gitlab_url="https://gitlab.com",
            access_token="glpat-xxxxxxxxxxxx",
        )
        # This test will fail until we implement GitLabDataSourceProvider
        assert provider is not None
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider
        assert isinstance(provider, GitLabDataSourceProvider)

    def test_get_all_types(self):
        """Test getting all datasource types."""
        types = DataSourceFactory.get_all_types()
        assert DataSourceType.TEXT in types
        assert DataSourceType.URL in types
        assert DataSourceType.FILE in types
        assert DataSourceType.GITLAB in types


class TestTextDataSourceProvider:
    """Tests for text datasource provider."""

    @pytest.mark.asyncio
    async def test_fetch_content(self):
        """Test fetching text content."""
        provider = TextDataSourceProvider()
        content = await provider.fetch_content("This is inline text content.")

        assert content.content == "This is inline text content."
        assert content.source == "inline_text"
        assert content.metadata["type"] == "inline"

    @pytest.mark.asyncio
    async def test_validate_source_valid(self):
        """Test validating valid text source."""
        provider = TextDataSourceProvider()
        is_valid = await provider.validate_source("Some text")
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_source_empty(self):
        """Test validating empty text source."""
        provider = TextDataSourceProvider()
        is_valid = await provider.validate_source("")
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_source_whitespace(self):
        """Test validating whitespace-only text source."""
        provider = TextDataSourceProvider()
        is_valid = await provider.validate_source("   ")
        assert is_valid is False


class TestURLDataSourceProvider:
    """Tests for URL datasource provider."""

    @pytest.mark.asyncio
    async def test_fetch_html_content(self):
        """Test fetching HTML content from URL."""
        provider = URLDataSourceProvider()

        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <nav>Navigation</nav>
                <main>Main content here</main>
                <footer>Footer</footer>
            </body>
        </html>
        """
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            content = await provider.fetch_content("https://example.com")

            assert "Main content here" in content.content
            assert content.metadata["title"] == "Test Page"
            # Nav and footer should be removed
            assert "Navigation" not in content.content
            assert "Footer" not in content.content

    @pytest.mark.asyncio
    async def test_validate_source_valid_url(self):
        """Test validating valid URL."""
        provider = URLDataSourceProvider()

        with patch("httpx.AsyncClient.head", new_callable=AsyncMock) as mock_head:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_head.return_value = mock_response

            is_valid = await provider.validate_source("https://example.com")
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_source_invalid_url(self):
        """Test validating invalid URL."""
        provider = URLDataSourceProvider()

        with patch("httpx.AsyncClient.head", new_callable=AsyncMock) as mock_head:
            mock_head.side_effect = httpx.RequestError("Connection failed")

            is_valid = await provider.validate_source("https://invalid.example")
            assert is_valid is False


class TestGitLabDataSourceProvider:
    """Tests for GitLab datasource provider."""

    def test_init(self):
        """Test GitLab provider initialization."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com",
            access_token="glpat-xxxxxxxxxxxx",
        )
        assert provider.gitlab_url == "https://gitlab.com"
        assert provider.access_token == "glpat-xxxxxxxxxxxx"

    def test_init_self_hosted(self):
        """Test GitLab provider with self-hosted instance."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.mycompany.com",
            access_token="glpat-xxxxxxxxxxxx",
        )
        assert provider.gitlab_url == "https://gitlab.mycompany.com"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from URL."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com/",
            access_token="glpat-xxxxxxxxxxxx",
        )
        assert provider.gitlab_url == "https://gitlab.com"

    @pytest.mark.asyncio
    async def test_fetch_file_content(self):
        """Test fetching a single file from GitLab repository."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com",
            access_token="glpat-xxxxxxxxxxxx",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "file_name": "README.md",
            "file_path": "README.md",
            "content": "IyBUZXN0IFJlcG9zaXRvcnk=",  # Base64 for "# Test Repository"
            "encoding": "base64",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            content = await provider.fetch_file(
                project_id="123",
                file_path="README.md",
                ref="main",
            )

            assert content.content == "# Test Repository"
            assert content.metadata["file_path"] == "README.md"
            mock_get.assert_called_once()
            # Verify correct headers
            call_kwargs = mock_get.call_args
            assert "PRIVATE-TOKEN" in call_kwargs.kwargs.get("headers", {})

    @pytest.mark.asyncio
    async def test_fetch_repository_tree(self):
        """Test fetching repository tree (file listing)."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com",
            access_token="glpat-xxxxxxxxxxxx",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "1", "name": "README.md", "type": "blob", "path": "README.md"},
            {"id": "2", "name": "src", "type": "tree", "path": "src"},
            {"id": "3", "name": "main.py", "type": "blob", "path": "src/main.py"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            tree = await provider.fetch_tree(
                project_id="123",
                path="",
                ref="main",
                recursive=True,
            )

            assert len(tree) == 3
            assert tree[0]["name"] == "README.md"
            assert tree[1]["type"] == "tree"

    @pytest.mark.asyncio
    async def test_fetch_issues(self):
        """Test fetching project issues."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com",
            access_token="glpat-xxxxxxxxxxxx",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 1,
                "iid": 1,
                "title": "Bug: Something broken",
                "description": "Details about the bug",
                "state": "opened",
                "labels": ["bug"],
            },
            {
                "id": 2,
                "iid": 2,
                "title": "Feature: New feature",
                "description": "Details about feature",
                "state": "opened",
                "labels": ["enhancement"],
            },
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            content = await provider.fetch_issues(
                project_id="123",
                state="opened",
            )

            assert "Bug: Something broken" in content.content
            assert "Feature: New feature" in content.content
            assert content.metadata["issue_count"] == 2

    @pytest.mark.asyncio
    async def test_fetch_merge_requests(self):
        """Test fetching merge requests."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com",
            access_token="glpat-xxxxxxxxxxxx",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 1,
                "iid": 10,
                "title": "Add new feature",
                "description": "This MR adds...",
                "state": "merged",
                "source_branch": "feature-branch",
                "target_branch": "main",
            },
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            content = await provider.fetch_merge_requests(
                project_id="123",
                state="merged",
            )

            assert "Add new feature" in content.content
            assert content.metadata["mr_count"] == 1

    @pytest.mark.asyncio
    async def test_fetch_content_dispatches_correctly(self):
        """Test that fetch_content dispatches to correct method based on source format."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com",
            access_token="glpat-xxxxxxxxxxxx",
        )

        # Test file source format: "project:123/file:README.md@main"
        with patch.object(provider, "fetch_file", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = DataSourceContent(content="test", source="test")
            await provider.fetch_content("project:123/file:README.md@main")
            mock_fetch.assert_called_once_with(
                project_id="123",
                file_path="README.md",
                ref="main",
            )

        # Test issues source format: "project:123/issues"
        with patch.object(provider, "fetch_issues", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = DataSourceContent(content="test", source="test")
            await provider.fetch_content("project:123/issues")
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_source_with_valid_token(self):
        """Test validating source with valid token."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com",
            access_token="glpat-xxxxxxxxxxxx",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 123, "name": "Test Project"}

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            is_valid = await provider.validate_source("project:123/file:README.md@main")
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_source_with_invalid_token(self):
        """Test validating source with invalid token."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com",
            access_token="invalid-token",
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(),
                response=MagicMock(status_code=401),
            )

            is_valid = await provider.validate_source("project:123/file:README.md@main")
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_fetch_multiple_files(self):
        """Test fetching multiple files from a repository."""
        from app.providers.datasource.gitlab_provider import GitLabDataSourceProvider

        provider = GitLabDataSourceProvider(
            gitlab_url="https://gitlab.com",
            access_token="glpat-xxxxxxxxxxxx",
        )

        with patch.object(provider, "fetch_tree", new_callable=AsyncMock) as mock_tree:
            mock_tree.return_value = [
                {"path": "README.md", "type": "blob"},
                {"path": "src/main.py", "type": "blob"},
            ]

            with patch.object(provider, "fetch_file", new_callable=AsyncMock) as mock_file:
                mock_file.side_effect = [
                    DataSourceContent(content="# README", source="README.md"),
                    DataSourceContent(content="print('hello')", source="src/main.py"),
                ]

                content = await provider.fetch_repository(
                    project_id="123",
                    ref="main",
                    file_patterns=["*.md", "*.py"],
                )

                assert "# README" in content.content
                assert "print('hello')" in content.content
                assert content.metadata["file_count"] == 2
