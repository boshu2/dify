"""
Unit tests for database migrations.
Tests Alembic configuration and migration scripts.
"""
import os
import pytest
from pathlib import Path
from alembic.config import Config
from alembic.script import ScriptDirectory


class TestAlembicConfiguration:
    """Tests for Alembic configuration."""

    @pytest.fixture
    def alembic_config(self):
        """Get Alembic configuration."""
        config_path = Path(__file__).parent.parent.parent / "alembic.ini"
        return Config(str(config_path))

    @pytest.fixture
    def script_dir(self, alembic_config):
        """Get Alembic script directory."""
        return ScriptDirectory.from_config(alembic_config)

    def test_alembic_config_exists(self):
        """Test that alembic.ini exists."""
        config_path = Path(__file__).parent.parent.parent / "alembic.ini"
        assert config_path.exists()

    def test_migrations_directory_exists(self):
        """Test that migrations directory exists."""
        migrations_path = Path(__file__).parent.parent.parent / "migrations"
        assert migrations_path.exists()
        assert migrations_path.is_dir()

    def test_versions_directory_exists(self):
        """Test that versions directory exists."""
        versions_path = Path(__file__).parent.parent.parent / "migrations" / "versions"
        assert versions_path.exists()
        assert versions_path.is_dir()

    def test_env_py_exists(self):
        """Test that env.py exists."""
        env_path = Path(__file__).parent.parent.parent / "migrations" / "env.py"
        assert env_path.exists()

    def test_script_location(self, alembic_config):
        """Test script location is correctly configured."""
        script_location = alembic_config.get_main_option("script_location")
        assert script_location == "migrations"


class TestMigrationScripts:
    """Tests for migration scripts."""

    @pytest.fixture
    def alembic_config(self):
        """Get Alembic configuration."""
        config_path = Path(__file__).parent.parent.parent / "alembic.ini"
        return Config(str(config_path))

    @pytest.fixture
    def script_dir(self, alembic_config):
        """Get Alembic script directory."""
        return ScriptDirectory.from_config(alembic_config)

    def test_has_initial_migration(self, script_dir):
        """Test that initial migration exists."""
        revisions = list(script_dir.walk_revisions())
        assert len(revisions) > 0

    def test_initial_migration_has_no_down_revision(self, script_dir):
        """Test that initial migration has no down_revision."""
        head = script_dir.get_current_head()
        assert head is not None

        # Walk to find base (the initial migration)
        revisions = list(script_dir.walk_revisions())
        base_revision = revisions[-1]  # Last one when walking from head
        assert base_revision.down_revision is None

    def test_migration_chain_is_valid(self, script_dir):
        """Test that migration chain is valid (no gaps)."""
        head = script_dir.get_current_head()
        assert head is not None

        # Walk from head to base should succeed
        revisions = list(script_dir.walk_revisions())
        assert len(revisions) >= 1

    def test_all_migrations_have_upgrade_downgrade(self, script_dir):
        """Test all migrations have upgrade and downgrade functions."""
        for revision in script_dir.walk_revisions():
            module = revision.module
            assert hasattr(module, "upgrade")
            assert hasattr(module, "downgrade")
            assert callable(module.upgrade)
            assert callable(module.downgrade)


class TestInitialMigration:
    """Tests for the initial schema migration."""

    @pytest.fixture
    def alembic_config(self):
        """Get Alembic configuration."""
        config_path = Path(__file__).parent.parent.parent / "alembic.ini"
        return Config(str(config_path))

    @pytest.fixture
    def script_dir(self, alembic_config):
        """Get Alembic script directory."""
        return ScriptDirectory.from_config(alembic_config)

    def test_initial_migration_revision_id(self, script_dir):
        """Test initial migration has expected revision ID."""
        revisions = list(script_dir.walk_revisions())
        initial = revisions[-1]
        assert initial.revision == "0001"

    def test_initial_migration_creates_providers_table(self, script_dir):
        """Test initial migration creates providers table."""
        revisions = list(script_dir.walk_revisions())
        initial = revisions[-1]

        # Read the migration file content
        migration_path = Path(initial.module.__file__)
        content = migration_path.read_text()

        assert "providers" in content
        assert "provider_type" in content
        assert "api_key" in content

    def test_initial_migration_creates_datasources_table(self, script_dir):
        """Test initial migration creates datasources table."""
        revisions = list(script_dir.walk_revisions())
        initial = revisions[-1]

        migration_path = Path(initial.module.__file__)
        content = migration_path.read_text()

        assert "datasources" in content
        assert "source_type" in content
        assert "gitlab_token" in content

    def test_initial_migration_creates_agents_table(self, script_dir):
        """Test initial migration creates agents table."""
        revisions = list(script_dir.walk_revisions())
        initial = revisions[-1]

        migration_path = Path(initial.module.__file__)
        content = migration_path.read_text()

        assert "agents" in content
        assert "system_prompt" in content
        assert "provider_id" in content

    def test_initial_migration_creates_association_table(self, script_dir):
        """Test initial migration creates agent_datasources table."""
        revisions = list(script_dir.walk_revisions())
        initial = revisions[-1]

        migration_path = Path(initial.module.__file__)
        content = migration_path.read_text()

        assert "agent_datasources" in content
