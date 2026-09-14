"""enable_rls_on_all_public_tables

Revision ID: a1b2c3d4e5f6
Revises: dfa570ee8243
Create Date: 2026-04-12 00:00:00.000000

Security fix: Enable Row-Level Security on all public-schema tables to
prevent unauthorized access via Supabase PostgREST (anon/authenticated
roles).  The backend connects as the `postgres` role which bypasses RLS,
so this migration has zero impact on application behaviour.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "dfa570ee8243"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Every regular table created by prior migrations.
# usage_daily_summary is a materialized view and does not support RLS.
_TABLES = [
    "analysis_history",
    "model_performance",
    "cost_tracking",
    "analytics_events",
    "api_keys",
    "credits",
    "transactions",
]


def upgrade() -> None:
    """Enable RLS and deny all access by default (no permissive policies)."""
    for table in _TABLES:
        op.execute(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE public.{table} FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    """Disable RLS on all tables (reverts to wide-open access)."""
    for table in _TABLES:
        op.execute(f"ALTER TABLE public.{table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY")
