"""add app permission tables

Revision ID: add_app_permission_tables
Revises: e8446f481c1e
Create Date: 2025-12-10 00:01:00.000000

"""
from alembic import op
import models as models
import sqlalchemy as sa


def _is_pg(conn):
    return conn.dialect.name == "postgresql"


# revision identifiers, used by Alembic.
revision = 'add_app_permission_tables'
down_revision = 'e8446f481c1e'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    # Create app_permissions table
    if _is_pg(conn):
        op.create_table(
            'app_permissions',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('account_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('role', sa.String(length=32), server_default=sa.text("'viewer'::character varying"), nullable=False),
            sa.Column('granted_by', models.types.StringUUID(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='app_permission_pkey'),
            sa.UniqueConstraint('app_id', 'account_id', name='unique_app_permission_app_account'),
        )
    else:
        op.create_table(
            'app_permissions',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('account_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('role', sa.String(length=32), server_default=sa.text("'viewer'"), nullable=False),
            sa.Column('granted_by', models.types.StringUUID(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='app_permission_pkey'),
            sa.UniqueConstraint('app_id', 'account_id', name='unique_app_permission_app_account'),
        )

    # Create indexes for app_permissions
    op.create_index('idx_app_permissions_app_id', 'app_permissions', ['app_id'])
    op.create_index('idx_app_permissions_account_id', 'app_permissions', ['account_id'])
    op.create_index('idx_app_permissions_tenant_id', 'app_permissions', ['tenant_id'])

    # Create app_access_configs table
    if _is_pg(conn):
        op.create_table(
            'app_access_configs',
            sa.Column('id', models.types.StringUUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('permission_type', sa.String(length=32), server_default=sa.text("'inherit_workspace'::character varying"), nullable=False),
            sa.Column('require_api_scope', sa.Boolean(), server_default=sa.text('false'), nullable=False),
            sa.Column('custom_rate_limit_rpm', sa.Integer(), nullable=True),
            sa.Column('custom_rate_limit_rph', sa.Integer(), nullable=True),
            sa.Column('access_description', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.PrimaryKeyConstraint('id', name='app_access_config_pkey'),
            sa.UniqueConstraint('app_id', name='unique_app_access_config_app_id'),
        )
    else:
        op.create_table(
            'app_access_configs',
            sa.Column('id', models.types.StringUUID(), nullable=False),
            sa.Column('app_id', models.types.StringUUID(), nullable=False),
            sa.Column('tenant_id', models.types.StringUUID(), nullable=False),
            sa.Column('permission_type', sa.String(length=32), server_default=sa.text("'inherit_workspace'"), nullable=False),
            sa.Column('require_api_scope', sa.Boolean(), server_default=sa.text('0'), nullable=False),
            sa.Column('custom_rate_limit_rpm', sa.Integer(), nullable=True),
            sa.Column('custom_rate_limit_rph', sa.Integer(), nullable=True),
            sa.Column('access_description', models.types.LongText(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
            sa.PrimaryKeyConstraint('id', name='app_access_config_pkey'),
            sa.UniqueConstraint('app_id', name='unique_app_access_config_app_id'),
        )

    # Create index for app_access_configs
    op.create_index('idx_app_access_configs_app_id', 'app_access_configs', ['app_id'])


def downgrade():
    op.drop_index('idx_app_access_configs_app_id', table_name='app_access_configs')
    op.drop_table('app_access_configs')
    op.drop_index('idx_app_permissions_tenant_id', table_name='app_permissions')
    op.drop_index('idx_app_permissions_account_id', table_name='app_permissions')
    op.drop_index('idx_app_permissions_app_id', table_name='app_permissions')
    op.drop_table('app_permissions')
