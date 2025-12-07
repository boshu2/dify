"""
GitLab data source provider.
Fetches files, issues, and merge requests from GitLab repositories.
"""
import base64
import fnmatch
import re
from typing import Any

import httpx

from app.providers.datasource.base import BaseDataSourceProvider, DataSourceContent


class GitLabDataSourceProvider(BaseDataSourceProvider):
    """
    GitLab API provider for fetching repository content.

    Supports:
    - Single file fetching
    - Repository tree listing
    - Multiple file fetching with patterns
    - Issue fetching
    - Merge request fetching

    Source path formats:
    - "project:123/file:README.md@main" - Single file
    - "project:123/tree:src@main" - Directory tree
    - "project:123/issues" - All issues
    - "project:123/issues?state=opened" - Filtered issues
    - "project:123/merge_requests" - All MRs
    - "project:123/repo@main" - Full repository (filtered by patterns)
    """

    def __init__(
        self,
        gitlab_url: str = "https://gitlab.com",
        access_token: str = "",
        timeout: float = 30.0,
    ):
        self.gitlab_url = gitlab_url.rstrip("/")
        self.access_token = access_token
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        """Get authentication headers."""
        return {
            "PRIVATE-TOKEN": self.access_token,
            "Accept": "application/json",
        }

    def _api_url(self, endpoint: str) -> str:
        """Build full API URL."""
        return f"{self.gitlab_url}/api/v4{endpoint}"

    async def fetch_content(self, source: str) -> DataSourceContent:
        """
        Fetch content based on source path format.

        Supported formats:
        - project:ID/file:PATH@REF
        - project:ID/tree:PATH@REF
        - project:ID/issues
        - project:ID/merge_requests
        - project:ID/repo@REF
        """
        # Parse the source path
        if "/file:" in source:
            match = re.match(r"project:([^/]+)/file:([^@]+)@(.+)", source)
            if match:
                return await self.fetch_file(
                    project_id=match.group(1),
                    file_path=match.group(2),
                    ref=match.group(3),
                )

        if "/issues" in source:
            match = re.match(r"project:([^/]+)/issues(?:\?(.+))?", source)
            if match:
                project_id = match.group(1)
                params = self._parse_query_params(match.group(2) or "")
                return await self.fetch_issues(
                    project_id=project_id,
                    **params,
                )

        if "/merge_requests" in source:
            match = re.match(r"project:([^/]+)/merge_requests(?:\?(.+))?", source)
            if match:
                project_id = match.group(1)
                params = self._parse_query_params(match.group(2) or "")
                return await self.fetch_merge_requests(
                    project_id=project_id,
                    **params,
                )

        if "/repo@" in source:
            match = re.match(r"project:([^/]+)/repo@(.+)", source)
            if match:
                return await self.fetch_repository(
                    project_id=match.group(1),
                    ref=match.group(2),
                )

        raise ValueError(f"Invalid GitLab source path format: {source}")

    def _parse_query_params(self, query: str) -> dict[str, str]:
        """Parse query string into dict."""
        if not query:
            return {}
        params = {}
        for pair in query.split("&"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                params[key] = value
        return params

    async def validate_source(self, source: str) -> bool:
        """Validate that the source is accessible with the given token."""
        try:
            # Extract project ID from source
            match = re.match(r"project:([^/]+)", source)
            if not match:
                return False

            project_id = match.group(1)

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self._api_url(f"/projects/{project_id}"),
                    headers=self._get_headers(),
                )
                return response.status_code == 200

        except Exception:
            return False

    async def fetch_file(
        self,
        project_id: str,
        file_path: str,
        ref: str = "main",
    ) -> DataSourceContent:
        """Fetch a single file from a GitLab repository."""
        # URL-encode the file path
        encoded_path = file_path.replace("/", "%2F")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                self._api_url(f"/projects/{project_id}/repository/files/{encoded_path}"),
                headers=self._get_headers(),
                params={"ref": ref},
            )
            response.raise_for_status()
            data = response.json()

        # Decode base64 content
        content = data.get("content", "")
        encoding = data.get("encoding", "base64")

        if encoding == "base64":
            content = base64.b64decode(content).decode("utf-8")

        return DataSourceContent(
            content=content,
            source=f"{self.gitlab_url}/{project_id}/-/blob/{ref}/{file_path}",
            metadata={
                "file_path": data.get("file_path", file_path),
                "file_name": data.get("file_name", file_path.split("/")[-1]),
                "ref": ref,
                "project_id": project_id,
            },
        )

    async def fetch_tree(
        self,
        project_id: str,
        path: str = "",
        ref: str = "main",
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch repository tree (file listing)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            params = {
                "ref": ref,
                "recursive": str(recursive).lower(),
                "per_page": 100,
            }
            if path:
                params["path"] = path

            response = await client.get(
                self._api_url(f"/projects/{project_id}/repository/tree"),
                headers=self._get_headers(),
                params=params,
            )
            response.raise_for_status()
            return response.json()

    async def fetch_issues(
        self,
        project_id: str,
        state: str = "all",
        labels: str | None = None,
        per_page: int = 100,
    ) -> DataSourceContent:
        """Fetch project issues."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            params = {
                "state": state,
                "per_page": per_page,
            }
            if labels:
                params["labels"] = labels

            response = await client.get(
                self._api_url(f"/projects/{project_id}/issues"),
                headers=self._get_headers(),
                params=params,
            )
            response.raise_for_status()
            issues = response.json()

        # Format issues as readable text
        lines = []
        for issue in issues:
            lines.append(f"## Issue #{issue['iid']}: {issue['title']}")
            lines.append(f"State: {issue['state']}")
            if issue.get("labels"):
                lines.append(f"Labels: {', '.join(issue['labels'])}")
            if issue.get("description"):
                lines.append(f"\n{issue['description']}")
            lines.append("\n---\n")

        content = "\n".join(lines)

        return DataSourceContent(
            content=content,
            source=f"{self.gitlab_url}/{project_id}/-/issues",
            metadata={
                "project_id": project_id,
                "issue_count": len(issues),
                "state_filter": state,
            },
        )

    async def fetch_merge_requests(
        self,
        project_id: str,
        state: str = "all",
        per_page: int = 100,
    ) -> DataSourceContent:
        """Fetch project merge requests."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                self._api_url(f"/projects/{project_id}/merge_requests"),
                headers=self._get_headers(),
                params={
                    "state": state,
                    "per_page": per_page,
                },
            )
            response.raise_for_status()
            mrs = response.json()

        # Format MRs as readable text
        lines = []
        for mr in mrs:
            lines.append(f"## MR !{mr['iid']}: {mr['title']}")
            lines.append(f"State: {mr['state']}")
            lines.append(f"Source: {mr.get('source_branch')} -> {mr.get('target_branch')}")
            if mr.get("description"):
                lines.append(f"\n{mr['description']}")
            lines.append("\n---\n")

        content = "\n".join(lines)

        return DataSourceContent(
            content=content,
            source=f"{self.gitlab_url}/{project_id}/-/merge_requests",
            metadata={
                "project_id": project_id,
                "mr_count": len(mrs),
                "state_filter": state,
            },
        )

    async def fetch_repository(
        self,
        project_id: str,
        ref: str = "main",
        file_patterns: list[str] | None = None,
        max_files: int = 50,
    ) -> DataSourceContent:
        """
        Fetch multiple files from a repository.

        Args:
            project_id: GitLab project ID
            ref: Git reference (branch, tag, commit)
            file_patterns: Glob patterns to filter files (e.g., ["*.md", "*.py"])
            max_files: Maximum number of files to fetch
        """
        # Get repository tree
        tree = await self.fetch_tree(project_id, "", ref, recursive=True)

        # Filter to only files (blobs)
        files = [item for item in tree if item["type"] == "blob"]

        # Apply pattern filtering if specified
        if file_patterns:
            filtered_files = []
            for f in files:
                for pattern in file_patterns:
                    if fnmatch.fnmatch(f["path"], pattern):
                        filtered_files.append(f)
                        break
            files = filtered_files

        # Limit number of files
        files = files[:max_files]

        # Fetch each file
        all_content = []
        for file_info in files:
            try:
                file_content = await self.fetch_file(
                    project_id=project_id,
                    file_path=file_info["path"],
                    ref=ref,
                )
                all_content.append(f"### {file_info['path']}\n```\n{file_content.content}\n```\n")
            except Exception:
                # Skip files that fail to fetch
                continue

        content = "\n".join(all_content)

        return DataSourceContent(
            content=content,
            source=f"{self.gitlab_url}/{project_id}/-/tree/{ref}",
            metadata={
                "project_id": project_id,
                "ref": ref,
                "file_count": len(all_content),
                "patterns": file_patterns,
            },
        )
