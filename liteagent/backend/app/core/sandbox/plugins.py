"""
Plugin system for LiteAgent sandbox.

Provides:
- Plugin discovery and installation
- Dependency management
- Plugin lifecycle management
- Tool registration from plugins
"""

import importlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx

from app.core.sandbox.config import SandboxConfig, get_sandbox_config


class PluginStatus(str, Enum):
    """Plugin installation status."""

    AVAILABLE = "available"
    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


class PluginType(str, Enum):
    """Types of plugins."""

    TOOL = "tool"
    PROVIDER = "provider"
    EXTRACTOR = "extractor"
    TRANSFORMER = "transformer"
    MIDDLEWARE = "middleware"


@dataclass
class PluginMetadata:
    """Plugin metadata."""

    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    plugin_type: PluginType = PluginType.TOOL
    homepage: str = ""
    repository: str = ""
    license: str = ""
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    python_requires: str = ">=3.11"
    entry_point: str = ""
    icon: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "python_requires": self.python_requires,
            "entry_point": self.entry_point,
            "icon": self.icon,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginMetadata":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            name=data["name"],
            version=data.get("version", "0.0.1"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            plugin_type=PluginType(data.get("plugin_type", "tool")),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            license=data.get("license", ""),
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            python_requires=data.get("python_requires", ">=3.11"),
            entry_point=data.get("entry_point", ""),
            icon=data.get("icon", ""),
        )


@dataclass
class Plugin:
    """Installed plugin."""

    metadata: PluginMetadata
    status: PluginStatus = PluginStatus.INSTALLED
    install_path: str = ""
    error_message: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    module: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "status": self.status.value,
            "install_path": self.install_path,
            "error_message": self.error_message,
            "config": self.config,
        }


class PluginInstallError(Exception):
    """Error during plugin installation."""
    pass


class PluginManager:
    """
    Manager for plugin installation and lifecycle.

    Supports:
    - Installing plugins from PyPI or git
    - Loading plugins from local directory
    - Enabling/disabling plugins
    - Plugin configuration
    - Tool registration from plugins
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or get_sandbox_config()
        self.plugins_dir = Path(self.config.plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        self._plugins: dict[str, Plugin] = {}
        self._tools: dict[str, Callable] = {}
        self._http_client: httpx.AsyncClient | None = None

        # Load existing plugins
        self._load_installed_plugins()

    def _load_installed_plugins(self) -> None:
        """Load plugins from plugins directory."""
        manifest_path = self.plugins_dir / "manifest.json"

        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)

                for plugin_data in manifest.get("plugins", []):
                    try:
                        metadata = PluginMetadata.from_dict(plugin_data["metadata"])
                        plugin = Plugin(
                            metadata=metadata,
                            status=PluginStatus(plugin_data.get("status", "installed")),
                            install_path=plugin_data.get("install_path", ""),
                            config=plugin_data.get("config", {}),
                        )
                        self._plugins[metadata.id] = plugin
                    except Exception as e:
                        print(f"Failed to load plugin: {e}")
            except Exception as e:
                print(f"Failed to load plugin manifest: {e}")

    def _save_manifest(self) -> None:
        """Save plugin manifest."""
        manifest_path = self.plugins_dir / "manifest.json"

        manifest = {
            "version": "1.0",
            "plugins": [p.to_dict() for p in self._plugins.values()],
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    async def search_plugins(
        self,
        query: str = "",
        plugin_type: PluginType | None = None,
        tags: list[str] | None = None,
    ) -> list[PluginMetadata]:
        """
        Search for plugins in registry.

        Args:
            query: Search query
            plugin_type: Filter by type
            tags: Filter by tags

        Returns:
            List of matching plugin metadata
        """
        if not self.config.plugin_registry_url:
            return []

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)

        try:
            params: dict[str, Any] = {"q": query}
            if plugin_type:
                params["type"] = plugin_type.value
            if tags:
                params["tags"] = ",".join(tags)

            response = await self._http_client.get(
                f"{self.config.plugin_registry_url}/api/plugins/search",
                params=params,
            )
            response.raise_for_status()

            data = response.json()
            return [PluginMetadata.from_dict(p) for p in data.get("plugins", [])]

        except Exception as e:
            print(f"Plugin search failed: {e}")
            return []

    async def install_from_pypi(
        self,
        package_name: str,
        version: str | None = None,
    ) -> Plugin:
        """
        Install a plugin from PyPI.

        Args:
            package_name: PyPI package name
            version: Specific version (optional)

        Returns:
            Installed plugin
        """
        # Build pip install command
        spec = f"{package_name}=={version}" if version else package_name

        try:
            # Install to plugins directory
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(self.plugins_dir / package_name),
                    spec,
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise PluginInstallError(f"pip install failed: {result.stderr}")

            # Try to load plugin metadata
            metadata = await self._load_plugin_metadata(package_name)

            plugin = Plugin(
                metadata=metadata,
                status=PluginStatus.INSTALLED,
                install_path=str(self.plugins_dir / package_name),
            )

            self._plugins[metadata.id] = plugin
            self._save_manifest()

            return plugin

        except subprocess.TimeoutExpired:
            raise PluginInstallError("Installation timed out")
        except Exception as e:
            raise PluginInstallError(f"Installation failed: {e}")

    async def install_from_git(
        self,
        repo_url: str,
        ref: str = "main",
    ) -> Plugin:
        """
        Install a plugin from git repository.

        Args:
            repo_url: Git repository URL
            ref: Branch, tag, or commit

        Returns:
            Installed plugin
        """
        # Extract repo name
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        install_path = self.plugins_dir / repo_name

        try:
            # Clone repository
            if install_path.exists():
                shutil.rmtree(install_path)

            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    ref,
                    repo_url,
                    str(install_path),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise PluginInstallError(f"git clone failed: {result.stderr}")

            # Install dependencies if requirements.txt exists
            requirements_path = install_path / "requirements.txt"
            if requirements_path.exists():
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(requirements_path),
                        "--target",
                        str(install_path / "deps"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode != 0:
                    print(f"Warning: Failed to install dependencies: {result.stderr}")

            # Load plugin metadata
            metadata = await self._load_plugin_metadata(repo_name)

            plugin = Plugin(
                metadata=metadata,
                status=PluginStatus.INSTALLED,
                install_path=str(install_path),
            )

            self._plugins[metadata.id] = plugin
            self._save_manifest()

            return plugin

        except subprocess.TimeoutExpired:
            raise PluginInstallError("Installation timed out")
        except Exception as e:
            raise PluginInstallError(f"Installation failed: {e}")

    async def install_from_directory(
        self,
        path: str | Path,
    ) -> Plugin:
        """
        Install a plugin from local directory.

        Args:
            path: Path to plugin directory

        Returns:
            Installed plugin
        """
        path = Path(path)

        if not path.exists():
            raise PluginInstallError(f"Directory not found: {path}")

        plugin_name = path.name
        install_path = self.plugins_dir / plugin_name

        try:
            # Copy to plugins directory
            if install_path.exists():
                shutil.rmtree(install_path)
            shutil.copytree(path, install_path)

            # Load metadata
            metadata = await self._load_plugin_metadata(plugin_name)

            plugin = Plugin(
                metadata=metadata,
                status=PluginStatus.INSTALLED,
                install_path=str(install_path),
            )

            self._plugins[metadata.id] = plugin
            self._save_manifest()

            return plugin

        except Exception as e:
            raise PluginInstallError(f"Installation failed: {e}")

    async def _load_plugin_metadata(self, plugin_name: str) -> PluginMetadata:
        """Load plugin metadata from plugin.json or pyproject.toml."""
        install_path = self.plugins_dir / plugin_name

        # Try plugin.json
        plugin_json = install_path / "plugin.json"
        if plugin_json.exists():
            with open(plugin_json) as f:
                data = json.load(f)
            return PluginMetadata.from_dict(data)

        # Try pyproject.toml
        pyproject = install_path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)

                project = data.get("project", {})
                tool = data.get("tool", {}).get("liteagent", {})

                return PluginMetadata(
                    id=tool.get("id", str(uuid4())),
                    name=project.get("name", plugin_name),
                    version=project.get("version", "0.0.1"),
                    description=project.get("description", ""),
                    author=", ".join(
                        a.get("name", "") for a in project.get("authors", [])
                    ),
                    plugin_type=PluginType(tool.get("type", "tool")),
                    dependencies=project.get("dependencies", []),
                    entry_point=tool.get("entry_point", ""),
                )
            except Exception:
                pass

        # Default metadata
        return PluginMetadata(
            id=str(uuid4()),
            name=plugin_name,
            version="0.0.1",
        )

    def uninstall(self, plugin_id: str) -> bool:
        """
        Uninstall a plugin.

        Args:
            plugin_id: Plugin ID

        Returns:
            True if uninstalled
        """
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False

        # Disable first
        self.disable(plugin_id)

        # Remove files
        if plugin.install_path:
            install_path = Path(plugin.install_path)
            if install_path.exists():
                shutil.rmtree(install_path)

        # Remove from registry
        del self._plugins[plugin_id]
        self._save_manifest()

        return True

    def enable(self, plugin_id: str) -> Plugin | None:
        """
        Enable a plugin.

        Args:
            plugin_id: Plugin ID

        Returns:
            Enabled plugin or None
        """
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return None

        try:
            # Load plugin module
            if plugin.install_path:
                sys.path.insert(0, plugin.install_path)

                # Try to import
                if plugin.metadata.entry_point:
                    module = importlib.import_module(plugin.metadata.entry_point)
                else:
                    # Try common patterns
                    for entry in ["main", plugin.metadata.name, "__init__"]:
                        try:
                            module = importlib.import_module(entry)
                            break
                        except ImportError:
                            continue
                    else:
                        raise ImportError(f"Could not find entry point for {plugin_id}")

                plugin.module = module

                # Register tools if plugin provides them
                if hasattr(module, "get_tools"):
                    tools = module.get_tools()
                    for tool in tools:
                        self._tools[tool.name] = tool

            plugin.status = PluginStatus.ENABLED
            self._save_manifest()

            return plugin

        except Exception as e:
            plugin.status = PluginStatus.ERROR
            plugin.error_message = str(e)
            self._save_manifest()
            return plugin

    def disable(self, plugin_id: str) -> Plugin | None:
        """
        Disable a plugin.

        Args:
            plugin_id: Plugin ID

        Returns:
            Disabled plugin or None
        """
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return None

        # Unregister tools
        if plugin.module and hasattr(plugin.module, "get_tools"):
            tools = plugin.module.get_tools()
            for tool in tools:
                self._tools.pop(tool.name, None)

        # Unload module
        plugin.module = None
        plugin.status = PluginStatus.DISABLED
        self._save_manifest()

        return plugin

    def get_plugin(self, plugin_id: str) -> Plugin | None:
        """Get a plugin by ID."""
        return self._plugins.get(plugin_id)

    def list_plugins(
        self,
        status: PluginStatus | None = None,
        plugin_type: PluginType | None = None,
    ) -> list[Plugin]:
        """List installed plugins."""
        plugins = list(self._plugins.values())

        if status:
            plugins = [p for p in plugins if p.status == status]
        if plugin_type:
            plugins = [p for p in plugins if p.metadata.plugin_type == plugin_type]

        return plugins

    def get_tools(self) -> dict[str, Callable]:
        """Get all tools from enabled plugins."""
        return self._tools.copy()

    def configure_plugin(
        self,
        plugin_id: str,
        config: dict[str, Any],
    ) -> Plugin | None:
        """
        Configure a plugin.

        Args:
            plugin_id: Plugin ID
            config: Configuration dict

        Returns:
            Configured plugin or None
        """
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return None

        plugin.config.update(config)
        self._save_manifest()

        # Reload if enabled
        if plugin.status == PluginStatus.ENABLED and plugin.module:
            if hasattr(plugin.module, "configure"):
                plugin.module.configure(plugin.config)

        return plugin

    async def close(self) -> None:
        """Close manager and release resources."""
        if self._http_client:
            await self._http_client.aclose()


# Global plugin manager
_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
