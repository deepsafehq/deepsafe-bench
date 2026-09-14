"""Add usage_daily_summary materialized view.

Revision ID: 6fb0f926b7a1
Revises: 248e159e13b5
Create Date: 2026-03-22 22:46:25.966565

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6fb0f926b7a1"
down_revision: Union[str, Sequence[str], None] = "248e159e13b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create usage_daily_summary materialized view for pre-aggregated daily stats."""
    op.execute(
        """
        CREATE MATERIALIZED VIEW usage_daily_summary AS
        WITH daily_base AS (
            SELECT
                ah.user_id,
                ah.timestamp::date AS date,
                COUNT(*) AS total_analyses,
                AVG(ah.confidence) AS avg_confidence
            FROM analysis_history ah
            WHERE ah.user_id IS NOT NULL
            GROUP BY ah.user_id, ah.timestamp::date
        ),
        media_counts AS (
            SELECT
                user_id,
                timestamp::date AS date,
                jsonb_object_agg(media_type, cnt) AS by_media_type
            FROM (
                SELECT user_id, timestamp::date, media_type, COUNT(*) AS cnt
                FROM analysis_history
                WHERE user_id IS NOT NULL
                GROUP BY user_id, timestamp::date, media_type
            ) sub
            GROUP BY user_id, date
        ),
        verdict_counts AS (
            SELECT
                user_id,
                timestamp::date AS date,
                jsonb_object_agg(verdict, cnt) AS verdicts
            FROM (
                SELECT user_id, timestamp::date, verdict, COUNT(*) AS cnt
                FROM analysis_history
                WHERE user_id IS NOT NULL
                GROUP BY user_id, timestamp::date, verdict
            ) sub
            GROUP BY user_id, date
        ),
        cost_agg AS (
            SELECT
                ct.user_id,
                ah.timestamp::date AS date,
                COALESCE(SUM(ct.credits_consumed), 0) AS total_credits_consumed,
                AVG(ct.total_inference_ms) AS avg_inference_ms
            FROM cost_tracking ct
            JOIN analysis_history ah ON ah.id = ct.analysis_id
            WHERE ct.user_id IS NOT NULL
            GROUP BY ct.user_id, ah.timestamp::date
        )
        SELECT
            db.user_id,
            db.date,
            db.total_analyses,
            COALESCE(mc.by_media_type, '{}'::jsonb) AS by_media_type,
            COALESCE(ca.total_credits_consumed, 0) AS total_credits_consumed,
            db.avg_confidence,
            ca.avg_inference_ms,
            COALESCE(vc.verdicts, '{}'::jsonb) AS verdicts
        FROM daily_base db
        LEFT JOIN media_counts mc ON mc.user_id = db.user_id AND mc.date = db.date
        LEFT JOIN verdict_counts vc ON vc.user_id = db.user_id AND vc.date = db.date
        LEFT JOIN cost_agg ca ON ca.user_id = db.user_id AND ca.date = db.date
        WITH DATA;
    """
    )

    op.execute(
        """
        CREATE UNIQUE INDEX idx_usage_daily_user_date
        ON usage_daily_summary (user_id, date);
    """
    )


def downgrade() -> None:
    """Drop usage_daily_summary materialized view."""
    op.execute("DROP MATERIALIZED VIEW IF EXISTS usage_daily_summary;")
