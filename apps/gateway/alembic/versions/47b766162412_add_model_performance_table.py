"""add model_performance table

Revision ID: 47b766162412
Revises: d5f124f36ddf
Create Date: 2026-03-22 22:42:19.643764

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "47b766162412"
down_revision: Union[str, Sequence[str], None] = "d5f124f36ddf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create model_performance table for per-model inference metrics."""
    op.create_table(
        "model_performance",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("model_score", sa.Float(), nullable=True),
        sa.Column("raw_output", sa.Text(), nullable=True),
        sa.Column("inference_time_ms", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["analysis_history.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_model_performance_analysis_id"),
        "model_performance",
        ["analysis_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_performance_user_id"),
        "model_performance",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop model_performance table."""
    op.drop_index(op.f("ix_model_performance_user_id"), table_name="model_performance")
    op.drop_index(
        op.f("ix_model_performance_analysis_id"), table_name="model_performance"
    )
    op.drop_table("model_performance")
