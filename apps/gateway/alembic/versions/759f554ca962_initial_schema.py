"""initial schema

Revision ID: 759f554ca962
Revises:
Create Date: 2026-03-22 22:28:21.187550

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "759f554ca962"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema: analysis_history, credits, and transactions tables."""
    op.create_table(
        "analysis_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=True),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("media_name", sa.String(), nullable=True),
        sa.Column("verdict", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("ensemble_method", sa.String(), nullable=False),
        sa.Column("ensemble_score", sa.Float(), nullable=False),
        sa.Column("inference_time", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("full_response", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_analysis_history_id"), "analysis_history", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_analysis_history_request_id"),
        "analysis_history",
        ["request_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_analysis_history_username"),
        "analysis_history",
        ["username"],
        unique=False,
    )

    op.create_table(
        "credits",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("balance", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_credits_id"), "credits", ["id"], unique=False)
    op.create_index(op.f("ix_credits_user_id"), "credits", ["user_id"], unique=True)

    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("stripe_session_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_transactions_id"), "transactions", ["id"], unique=False)
    op.create_index(
        op.f("ix_transactions_user_id"), "transactions", ["user_id"], unique=False
    )


def downgrade() -> None:
    """Drop all tables created in the initial schema."""
    op.drop_index(op.f("ix_transactions_user_id"), table_name="transactions")
    op.drop_index(op.f("ix_transactions_id"), table_name="transactions")
    op.drop_table("transactions")

    op.drop_index(op.f("ix_credits_user_id"), table_name="credits")
    op.drop_index(op.f("ix_credits_id"), table_name="credits")
    op.drop_table("credits")

    op.drop_index(op.f("ix_analysis_history_username"), table_name="analysis_history")
    op.drop_index(op.f("ix_analysis_history_request_id"), table_name="analysis_history")
    op.drop_index(op.f("ix_analysis_history_id"), table_name="analysis_history")
    op.drop_table("analysis_history")
