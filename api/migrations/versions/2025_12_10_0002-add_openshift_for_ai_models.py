"""add openshift for ai models - external consumer, api token scope, published app

Revision ID: add_openshift_for_ai_models
Revises: add_app_permission_tables
Create Date: 2025-12-10 00:02:00.000000

"""
from alembic import op
import models as models
import sqlalchemy as sa


def _is_pg(conn):
    return conn.dialect.name == "postgresql"


# revision identifiers, used by Alembic.
revision = 'add_openshift_for_ai_models'
down_revision = 'add_app_permission_tables'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    # ==========================================
    # External Consumer Tables
    # ==========================================

    # Create external_consumers table
    if _is_pg(conn):
        op.create_table(
            'external_consumers',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('email', sa.String(length=255), nullable=False),
            sa.Column('organization', sa.String(length=255), nullable=True),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('auth_type', sa.String(length=32), server_default=sa.text("'api_key'::character varying"), nullable=False),
            sa.Column('api_key_hash', sa.String(length=255), nullable=True),
            sa.Column('api_key_prefix', sa.String(length=16), nullable=True),
            sa.Column('status', sa.String(length=32), server_default=sa.text("'active'::character varying"), nullable=False),
            sa.Column('rate_limit_rpm', sa.Integer(), nullable=True),
            sa.Column('rate_limit_rph', sa.Integer(), nullable=True),
            sa.Column('rate_limit_rpd', sa.Integer(), nullable=True),
            sa.Column('quota_total', sa.Integer(), nullable=True),
            sa.Column('quota_used', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('quota_reset_at', sa.DateTime(), nullable=True),
            sa.Column('metadata_json', sa.Text(), nullable=True),
            sa.Column('webhook_url', sa.String(length=512), nullable=True),
            sa.Column('created_by', models.types.StringUUID(), nullable=False),
            sa.Column('last_active_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='external_consumer_pkey'),
            sa.UniqueConstraint('tenant_id', 'email', name='unique_external_consumer_email'),
        )
    else:
        op.create_table(
            'external_consumers',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('email', sa.String(length=255), nullable=False),
            sa.Column('organization', sa.String(length=255), nullable=True),
            sa.Column('description', models.types.LongText(), nullable=True),
            sa.Column('auth_type', sa.String(length=32), server_default=sa.text("'api_key'"), nullable=False),
            sa.Column('api_key_hash', sa.String(length=255), nullable=True),
            sa.Column('api_key_prefix', sa.String(length=16), nullable=True),
            sa.Column('status', sa.String(length=32), server_default=sa.text("'active'"), nullable=False),
            sa.Column('rate_limit_rpm', sa.Integer(), nullable=True),
            sa.Column('rate_limit_rph', sa.Integer(), nullable=True),
            sa.Column('rate_limit_rpd', sa.Integer(), nullable=True),
            sa.Column('quota_total', sa.Integer(), nullable=True),
            sa.Column('quota_used', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('quota_reset_at', sa.DateTime(), nullable=True),
            sa.Column('metadata_json', models.types.LongText(), nullable=True),
            sa.Column('webhook_url', sa.String(length=512), nullable=True),
            sa.Column('created_by', models.types.StringUUID(), nullable=False),
            sa.Column('last_active_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='external_consumer_pkey'),
            sa.UniqueConstraint('tenant_id', 'email', name='unique_external_consumer_email'),
        )

    op.create_index('idx_external_consumers_tenant_id', 'external_consumers', ['tenant_id'])
    op.create_index('idx_external_consumers_email', 'external_consumers', ['email'])
    op.create_index('idx_external_consumers_status', 'external_consumers', ['status'])

    # Create external_consumer_app_access table
    if _is_pg(conn):
        op.create_table(
            'external_consumer_app_access',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('consumer_id', models.types.StringUUID(), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('can_invoke', sa.Boolean(), server_default=sa.text('true'), nullable=False),
            sa.Column('can_view_logs', sa.Boolean(), server_default=sa.text('false'), nullable=False),
            sa.Column('custom_rate_limit_rpm', sa.Integer(), nullable=True),
            sa.Column('custom_rate_limit_rph', sa.Integer(), nullable=True),
            sa.Column('app_quota_total', sa.Integer(), nullable=True),
            sa.Column('app_quota_used', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('allowed_scopes', sa.Text(), nullable=True),
            sa.Column('valid_from', sa.DateTime(), nullable=True),
            sa.Column('valid_until', sa.DateTime(), nullable=True),
            sa.Column('granted_by', models.types.StringUUID(), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='external_consumer_app_access_pkey'),
            sa.UniqueConstraint('consumer_id', 'app_id', name='unique_consumer_app_access'),
        )
    else:
        op.create_table(
            'external_consumer_app_access',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('consumer_id', models.types.StringUUID(), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('can_invoke', sa.Boolean(), server_default=sa.text('1'), nullable=False),
            sa.Column('can_view_logs', sa.Boolean(), server_default=sa.text('0'), nullable=False),
            sa.Column('custom_rate_limit_rpm', sa.Integer(), nullable=True),
            sa.Column('custom_rate_limit_rph', sa.Integer(), nullable=True),
            sa.Column('app_quota_total', sa.Integer(), nullable=True),
            sa.Column('app_quota_used', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('allowed_scopes', models.types.LongText(), nullable=True),
            sa.Column('valid_from', sa.DateTime(), nullable=True),
            sa.Column('valid_until', sa.DateTime(), nullable=True),
            sa.Column('granted_by', models.types.StringUUID(), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='external_consumer_app_access_pkey'),
            sa.UniqueConstraint('consumer_id', 'app_id', name='unique_consumer_app_access'),
        )

    op.create_index('idx_external_consumer_app_access_consumer_id', 'external_consumer_app_access', ['consumer_id'])
    op.create_index('idx_external_consumer_app_access_app_id', 'external_consumer_app_access', ['app_id'])

    # Create external_consumer_usage_logs table
    if _is_pg(conn):
        op.create_table(
            'external_consumer_usage_logs',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('consumer_id', models.types.StringUUID(), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('endpoint', sa.String(length=255), nullable=False),
            sa.Column('method', sa.String(length=16), nullable=False),
            sa.Column('status_code', sa.Integer(), nullable=False),
            sa.Column('response_time_ms', sa.Integer(), nullable=False),
            sa.Column('prompt_tokens', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('completion_tokens', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('total_tokens', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('estimated_cost', sa.Float(), server_default=sa.text('0'), nullable=False),
            sa.Column('request_id', sa.String(length=64), nullable=True),
            sa.Column('ip_address', sa.String(length=64), nullable=True),
            sa.Column('user_agent', sa.String(length=512), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='external_consumer_usage_log_pkey'),
        )
    else:
        op.create_table(
            'external_consumer_usage_logs',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('consumer_id', models.types.StringUUID(), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('endpoint', sa.String(length=255), nullable=False),
            sa.Column('method', sa.String(length=16), nullable=False),
            sa.Column('status_code', sa.Integer(), nullable=False),
            sa.Column('response_time_ms', sa.Integer(), nullable=False),
            sa.Column('prompt_tokens', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('completion_tokens', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('total_tokens', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('estimated_cost', sa.Float(), server_default=sa.text('0'), nullable=False),
            sa.Column('request_id', sa.String(length=64), nullable=True),
            sa.Column('ip_address', sa.String(length=64), nullable=True),
            sa.Column('user_agent', sa.String(length=512), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='external_consumer_usage_log_pkey'),
        )

    op.create_index('idx_external_consumer_usage_logs_consumer_id', 'external_consumer_usage_logs', ['consumer_id'])
    op.create_index('idx_external_consumer_usage_logs_app_id', 'external_consumer_usage_logs', ['app_id'])
    op.create_index('idx_external_consumer_usage_logs_created_at', 'external_consumer_usage_logs', ['created_at'])

    # ==========================================
    # API Token Scope Tables
    # ==========================================

    # Create api_token_scopes table
    if _is_pg(conn):
        op.create_table(
            'api_token_scopes',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('token_id', models.types.StringUUID(), nullable=False),
            sa.Column('scope', sa.String(length=64), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='api_token_scope_pkey'),
            sa.UniqueConstraint('token_id', 'scope', name='unique_token_scope'),
        )
    else:
        op.create_table(
            'api_token_scopes',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('token_id', models.types.StringUUID(), nullable=False),
            sa.Column('scope', sa.String(length=64), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='api_token_scope_pkey'),
            sa.UniqueConstraint('token_id', 'scope', name='unique_token_scope'),
        )

    op.create_index('idx_api_token_scopes_token_id', 'api_token_scopes', ['token_id'])

    # Create api_token_configs table
    if _is_pg(conn):
        op.create_table(
            'api_token_configs',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('token_id', models.types.StringUUID(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=True),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('expires_at', sa.DateTime(), nullable=True),
            sa.Column('is_expired', sa.Boolean(), server_default=sa.text('false'), nullable=False),
            sa.Column('rate_limit_rpm', sa.Integer(), nullable=True),
            sa.Column('rate_limit_rph', sa.Integer(), nullable=True),
            sa.Column('total_requests', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('total_tokens_used', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('last_used_at', sa.DateTime(), nullable=True),
            sa.Column('last_used_ip', sa.String(length=64), nullable=True),
            sa.Column('allowed_ips', sa.Text(), nullable=True),
            sa.Column('allowed_referrers', sa.Text(), nullable=True),
            sa.Column('external_consumer_id', models.types.StringUUID(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='api_token_config_pkey'),
            sa.UniqueConstraint('token_id', name='unique_api_token_config_token_id'),
        )
    else:
        op.create_table(
            'api_token_configs',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('token_id', models.types.StringUUID(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=True),
            sa.Column('description', models.types.LongText(), nullable=True),
            sa.Column('expires_at', sa.DateTime(), nullable=True),
            sa.Column('is_expired', sa.Boolean(), server_default=sa.text('0'), nullable=False),
            sa.Column('rate_limit_rpm', sa.Integer(), nullable=True),
            sa.Column('rate_limit_rph', sa.Integer(), nullable=True),
            sa.Column('total_requests', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('total_tokens_used', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('last_used_at', sa.DateTime(), nullable=True),
            sa.Column('last_used_ip', sa.String(length=64), nullable=True),
            sa.Column('allowed_ips', models.types.LongText(), nullable=True),
            sa.Column('allowed_referrers', models.types.LongText(), nullable=True),
            sa.Column('external_consumer_id', models.types.StringUUID(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='api_token_config_pkey'),
            sa.UniqueConstraint('token_id', name='unique_api_token_config_token_id'),
        )

    op.create_index('idx_api_token_configs_token_id', 'api_token_configs', ['token_id'])

    # ==========================================
    # Published App Tables
    # ==========================================

    # Create published_apps table
    if _is_pg(conn):
        op.create_table(
            'published_apps',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('slug', sa.String(length=128), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('icon', sa.String(length=255), nullable=True),
            sa.Column('icon_background', sa.String(length=32), nullable=True),
            sa.Column('version', sa.String(length=32), nullable=False),
            sa.Column('changelog', sa.Text(), nullable=True),
            sa.Column('status', sa.String(length=32), server_default=sa.text("'draft'::character varying"), nullable=False),
            sa.Column('visibility', sa.String(length=32), server_default=sa.text("'private'::character varying"), nullable=False),
            sa.Column('default_rate_limit_rpm', sa.Integer(), server_default=sa.text('60'), nullable=False),
            sa.Column('default_rate_limit_rph', sa.Integer(), server_default=sa.text('1000'), nullable=False),
            sa.Column('default_rate_limit_rpd', sa.Integer(), nullable=True),
            sa.Column('free_quota_per_consumer', sa.Integer(), nullable=True),
            sa.Column('pricing_model', sa.String(length=32), nullable=True),
            sa.Column('price_per_request', sa.Float(), nullable=True),
            sa.Column('terms_of_service', sa.Text(), nullable=True),
            sa.Column('privacy_policy', sa.Text(), nullable=True),
            sa.Column('documentation_url', sa.String(length=512), nullable=True),
            sa.Column('support_email', sa.String(length=255), nullable=True),
            sa.Column('custom_domain', sa.String(length=255), nullable=True),
            sa.Column('cors_allowed_origins', sa.Text(), nullable=True),
            sa.Column('webhook_url', sa.String(length=512), nullable=True),
            sa.Column('total_consumers', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('total_requests', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('published_by', models.types.StringUUID(), nullable=True),
            sa.Column('published_at', sa.DateTime(), nullable=True),
            sa.Column('created_by', models.types.StringUUID(), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='published_app_pkey'),
            sa.UniqueConstraint('slug', name='unique_published_app_slug'),
            sa.UniqueConstraint('tenant_id', 'app_id', name='unique_published_app_per_tenant_app'),
        )
    else:
        op.create_table(
            'published_apps',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('slug', sa.String(length=128), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('description', models.types.LongText(), nullable=True),
            sa.Column('icon', sa.String(length=255), nullable=True),
            sa.Column('icon_background', sa.String(length=32), nullable=True),
            sa.Column('version', sa.String(length=32), nullable=False),
            sa.Column('changelog', models.types.LongText(), nullable=True),
            sa.Column('status', sa.String(length=32), server_default=sa.text("'draft'"), nullable=False),
            sa.Column('visibility', sa.String(length=32), server_default=sa.text("'private'"), nullable=False),
            sa.Column('default_rate_limit_rpm', sa.Integer(), server_default=sa.text('60'), nullable=False),
            sa.Column('default_rate_limit_rph', sa.Integer(), server_default=sa.text('1000'), nullable=False),
            sa.Column('default_rate_limit_rpd', sa.Integer(), nullable=True),
            sa.Column('free_quota_per_consumer', sa.Integer(), nullable=True),
            sa.Column('pricing_model', sa.String(length=32), nullable=True),
            sa.Column('price_per_request', sa.Float(), nullable=True),
            sa.Column('terms_of_service', models.types.LongText(), nullable=True),
            sa.Column('privacy_policy', models.types.LongText(), nullable=True),
            sa.Column('documentation_url', sa.String(length=512), nullable=True),
            sa.Column('support_email', sa.String(length=255), nullable=True),
            sa.Column('custom_domain', sa.String(length=255), nullable=True),
            sa.Column('cors_allowed_origins', models.types.LongText(), nullable=True),
            sa.Column('webhook_url', sa.String(length=512), nullable=True),
            sa.Column('total_consumers', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('total_requests', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('published_by', models.types.StringUUID(), nullable=True),
            sa.Column('published_at', sa.DateTime(), nullable=True),
            sa.Column('created_by', models.types.StringUUID(), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='published_app_pkey'),
            sa.UniqueConstraint('slug', name='unique_published_app_slug'),
            sa.UniqueConstraint('tenant_id', 'app_id', name='unique_published_app_per_tenant_app'),
        )

    op.create_index('idx_published_apps_tenant_id', 'published_apps', ['tenant_id'])
    op.create_index('idx_published_apps_app_id', 'published_apps', ['app_id'])
    op.create_index('idx_published_apps_slug', 'published_apps', ['slug'])
    op.create_index('idx_published_apps_status', 'published_apps', ['status'])

    # Create published_app_versions table
    if _is_pg(conn):
        op.create_table(
            'published_app_versions',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('published_app_id', models.types.StringUUID(), nullable=False),
            sa.Column('version', sa.String(length=32), nullable=False),
            sa.Column('workflow_version', sa.String(length=255), nullable=True),
            sa.Column('config_snapshot', sa.Text(), nullable=True),
            sa.Column('changelog', sa.Text(), nullable=True),
            sa.Column('is_current', sa.Boolean(), server_default=sa.text('false'), nullable=False),
            sa.Column('is_deprecated', sa.Boolean(), server_default=sa.text('false'), nullable=False),
            sa.Column('deprecation_message', sa.Text(), nullable=True),
            sa.Column('created_by', models.types.StringUUID(), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='published_app_version_pkey'),
            sa.UniqueConstraint('published_app_id', 'version', name='unique_published_app_version'),
        )
    else:
        op.create_table(
            'published_app_versions',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('published_app_id', models.types.StringUUID(), nullable=False),
            sa.Column('version', sa.String(length=32), nullable=False),
            sa.Column('workflow_version', sa.String(length=255), nullable=True),
            sa.Column('config_snapshot', models.types.LongText(), nullable=True),
            sa.Column('changelog', models.types.LongText(), nullable=True),
            sa.Column('is_current', sa.Boolean(), server_default=sa.text('0'), nullable=False),
            sa.Column('is_deprecated', sa.Boolean(), server_default=sa.text('0'), nullable=False),
            sa.Column('deprecation_message', models.types.LongText(), nullable=True),
            sa.Column('created_by', models.types.StringUUID(), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='published_app_version_pkey'),
            sa.UniqueConstraint('published_app_id', 'version', name='unique_published_app_version'),
        )

    op.create_index('idx_published_app_versions_published_app_id', 'published_app_versions', ['published_app_id'])
    op.create_index('idx_published_app_versions_version', 'published_app_versions', ['version'])

    # Create published_app_webhooks table
    if _is_pg(conn):
        op.create_table(
            'published_app_webhooks',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('published_app_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('url', sa.String(length=512), nullable=False),
            sa.Column('secret', sa.String(length=255), nullable=True),
            sa.Column('events', sa.Text(), nullable=False),
            sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
            sa.Column('consecutive_failures', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('last_failure_at', sa.DateTime(), nullable=True),
            sa.Column('last_failure_reason', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='published_app_webhook_pkey'),
        )
    else:
        op.create_table(
            'published_app_webhooks',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('published_app_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('url', sa.String(length=512), nullable=False),
            sa.Column('secret', sa.String(length=255), nullable=True),
            sa.Column('events', models.types.LongText(), nullable=False),
            sa.Column('is_active', sa.Boolean(), server_default=sa.text('1'), nullable=False),
            sa.Column('consecutive_failures', sa.Integer(), server_default=sa.text('0'), nullable=False),
            sa.Column('last_failure_at', sa.DateTime(), nullable=True),
            sa.Column('last_failure_reason', models.types.LongText(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='published_app_webhook_pkey'),
        )

    op.create_index('idx_published_app_webhooks_published_app_id', 'published_app_webhooks', ['published_app_id'])


def downgrade():
    # Drop published app tables
    op.drop_index('idx_published_app_webhooks_published_app_id', table_name='published_app_webhooks')
    op.drop_table('published_app_webhooks')
    op.drop_index('idx_published_app_versions_version', table_name='published_app_versions')
    op.drop_index('idx_published_app_versions_published_app_id', table_name='published_app_versions')
    op.drop_table('published_app_versions')
    op.drop_index('idx_published_apps_status', table_name='published_apps')
    op.drop_index('idx_published_apps_slug', table_name='published_apps')
    op.drop_index('idx_published_apps_app_id', table_name='published_apps')
    op.drop_index('idx_published_apps_tenant_id', table_name='published_apps')
    op.drop_table('published_apps')

    # Drop API token scope tables
    op.drop_index('idx_api_token_configs_token_id', table_name='api_token_configs')
    op.drop_table('api_token_configs')
    op.drop_index('idx_api_token_scopes_token_id', table_name='api_token_scopes')
    op.drop_table('api_token_scopes')

    # Drop external consumer tables
    op.drop_index('idx_external_consumer_usage_logs_created_at', table_name='external_consumer_usage_logs')
    op.drop_index('idx_external_consumer_usage_logs_app_id', table_name='external_consumer_usage_logs')
    op.drop_index('idx_external_consumer_usage_logs_consumer_id', table_name='external_consumer_usage_logs')
    op.drop_table('external_consumer_usage_logs')
    op.drop_index('idx_external_consumer_app_access_app_id', table_name='external_consumer_app_access')
    op.drop_index('idx_external_consumer_app_access_consumer_id', table_name='external_consumer_app_access')
    op.drop_table('external_consumer_app_access')
    op.drop_index('idx_external_consumers_status', table_name='external_consumers')
    op.drop_index('idx_external_consumers_email', table_name='external_consumers')
    op.drop_index('idx_external_consumers_tenant_id', table_name='external_consumers')
    op.drop_table('external_consumers')
