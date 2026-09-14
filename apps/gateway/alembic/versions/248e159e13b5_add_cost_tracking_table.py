"""add cost_tracking table

Revision ID: 248e159e13b5
Revises: 47b766162412
Create Date: 2026-03-22 22:42:30.776427

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "248e159e13b5"
down_revision: Union[str, Sequence[str], None] = "47b766162412"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create cost_tracking table for per-analysis cost attribution."""
    op.create_table(
        "cost_tracking",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("credits_consumed", sa.Integer(), nullable=False),
        sa.Column("model_count", sa.Integer(), nullable=False),
        sa.Column("total_inference_ms", sa.Float(), nullable=True),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["analysis_history.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_cost_tracking_analysis_id"),
        "cost_tracking",
        ["analysis_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_cost_tracking_user_id"),
        "cost_tracking",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop cost_tracking table."""
    op.drop_index(op.f("ix_cost_tracking_user_id"), table_name="cost_tracking")
    op.drop_index(op.f("ix_cost_tracking_analysis_id"), table_name="cost_tracking")
    op.drop_table("cost_tracking")
