"""Tests for Postgres-backed analytics module."""

import json
from unittest.mock import MagicMock, patch

import analytics


def test_track_inserts_row():
    """track() should INSERT a row into analytics_events."""
    mock_session = MagicMock()
    mock_session_cls = MagicMock(return_value=mock_session)

    with patch.object(analytics, "_SessionLocal", mock_session_cls):
        with patch.object(analytics, "_initialized", True):
            analytics.track("user-123", "analysis_completed", {"media_type": "image"})

    mock_session.add.assert_called_once()
    event = mock_session.add.call_args[0][0]
    assert event.user_id == "user-123"
    assert event.event_name == "analysis_completed"
    assert json.loads(event.properties) == {"media_type": "image"}
    mock_session.commit.assert_called_once()
    mock_session.close.assert_called_once()


def test_track_does_not_raise_on_db_error():
    """track() must be fire-and-forget — never raises."""
    mock_session = MagicMock()
    mock_session.commit.side_effect = Exception("DB down")
    mock_session_cls = MagicMock(return_value=mock_session)

    with patch.object(analytics, "_SessionLocal", mock_session_cls):
        with patch.object(analytics, "_initialized", True):
            analytics.track("user-123", "test_event", {"key": "value"})

    mock_session.rollback.assert_called_once()
    mock_session.close.assert_called_once()


def test_track_skips_when_not_initialized():
    """track() should no-op when analytics is not initialized."""
    with patch.object(analytics, "_initialized", False):
        analytics.track("user-123", "test_event")


def test_track_accepts_none_user_id():
    """track() should accept None as user_id."""
    mock_session = MagicMock()
    mock_session_cls = MagicMock(return_value=mock_session)

    with patch.object(analytics, "_SessionLocal", mock_session_cls):
        with patch.object(analytics, "_initialized", True):
            analytics.track(None, "analysis_started", {"job_id": "abc"})

    event = mock_session.add.call_args[0][0]
    assert event.user_id is None


def test_identify_stores_as_event():
    """identify() should store a user_identified event."""
    mock_session = MagicMock()
    mock_session_cls = MagicMock(return_value=mock_session)

    with patch.object(analytics, "_SessionLocal", mock_session_cls):
        with patch.object(analytics, "_initialized", True):
            analytics.identify("user-123", {"email": "test@test.com"})

    event = mock_session.add.call_args[0][0]
    assert event.event_name == "user_identified"
    assert json.loads(event.properties) == {"email": "test@test.com"}
