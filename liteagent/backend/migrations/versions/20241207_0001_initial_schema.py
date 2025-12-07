"""Initial schema - providers, datasources, agents

Revision ID: 0001
Revises:
Create Date: 2024-12-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create providers table
    op.create_table(
        'providers',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('provider_type', sa.Enum('openai', 'anthropic', 'ollama', 'openai_compatible', name='providertype'), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('api_key', sa.Text(), nullable=True),
        sa.Column('base_url', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_providers_name', 'providers', ['name'])

    # Create datasources table
    op.create_table(
        'datasources',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('source_type', sa.Enum('file', 'url', 'text', 'gitlab', name='datasourcetype'), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('source_path', sa.String(1000), nullable=True),
        sa.Column('gitlab_url', sa.String(500), nullable=True),
        sa.Column('gitlab_token', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_datasources_name', 'datasources', ['name'])

    # Create agents table
    op.create_table(
        'agents',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('provider_id', sa.String(36), sa.ForeignKey('providers.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_agents_name', 'agents', ['name'])

    # Create agent_datasources association table
    op.create_table(
        'agent_datasources',
        sa.Column('agent_id', sa.String(36), sa.ForeignKey('agents.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('datasource_id', sa.String(36), sa.ForeignKey('datasources.id', ondelete='CASCADE'), primary_key=True),
    )


def downgrade() -> None:
    op.drop_table('agent_datasources')
    op.drop_table('agents')
    op.drop_table('datasources')
    op.drop_table('providers')

    # Drop enums (PostgreSQL specific, SQLite ignores this)
    op.execute("DROP TYPE IF EXISTS providertype")
    op.execute("DROP TYPE IF EXISTS datasourcetype")
