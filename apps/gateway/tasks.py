"""Celery periodic tasks for DeepSafe analytics."""

import logging

from celery_app import celery_app
from database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.refresh_usage_summary")
def refresh_usage_summary() -> None:
    """Refresh the usage_daily_summary materialized view.

    Executes a REFRESH MATERIALIZED VIEW CONCURRENTLY statement so that
    reads against the view remain available during the refresh.
    """
    try:
        with engine.connect() as conn:
            conn.execute(
                text("REFRESH MATERIALIZED VIEW CONCURRENTLY usage_daily_summary")
            )
            conn.commit()
        logger.info("Refreshed usage_daily_summary materialized view.")
    except Exception as exc:
        logger.error(f"Failed to refresh usage_daily_summary: {exc}")
