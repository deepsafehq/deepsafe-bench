"""Postgres-backed analytics for DeepSafe.

Fire-and-forget event tracking. Analytics failures never block
the detection critical path.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SessionLocal = None
_initialized = False


def init_analytics() -> None:
    """Initialize analytics with a SessionLocal factory.

    Call once at app startup. Imports SessionLocal from database
    module to avoid circular imports at module level.
    """
    global _SessionLocal, _initialized
    try:
        from database import SessionLocal

        _SessionLocal = SessionLocal
        _initialized = True
        logger.info("Postgres analytics initialized.")
    except Exception as exc:
        logger.warning(f"Analytics init failed: {exc}")
        _initialized = False


def track(
    distinct_id: Optional[str],
    event: str,
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture an analytics event. Fire-and-forget.

    Args:
        distinct_id: The unique identifier for the user (nullable).
        event: The name of the event to capture.
        properties: Optional dictionary of event properties.
    """
    if not _initialized or _SessionLocal is None:
        return

    session = _SessionLocal()
    try:
        from database import AnalyticsEvent

        row = AnalyticsEvent(
            user_id=distinct_id,
            event_name=event,
            properties=json.dumps(properties) if properties else None,
        )
        session.add(row)
        session.commit()
    except Exception as exc:
        logger.warning(f"Analytics track failed for '{event}': {exc}")
        session.rollback()
    finally:
        session.close()


def identify(distinct_id: str, properties: Optional[Dict[str, Any]] = None) -> None:
    """Identify a user with properties. Fire-and-forget.

    Stored as a 'user_identified' event.

    Args:
        distinct_id: The unique identifier for the user.
        properties: Optional dictionary of user properties.
    """
    track(distinct_id, "user_identified", properties)


def shutdown_analytics() -> None:
    """No-op. Retained for interface stability.

    Postgres writes are synchronous, so there is no queue to flush.
    """
    pass
