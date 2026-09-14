"""Tests for analytics fire-and-forget safety."""

from unittest.mock import MagicMock, patch

import analytics


def test_track_does_not_raise_on_db_error():
    """analytics.track must never raise, even if DB fails."""
    mock_session = MagicMock()
    mock_session.commit.side_effect = Exception("DB exploded")
    mock_session_cls = MagicMock(return_value=mock_session)

    with patch.object(analytics, "_SessionLocal", mock_session_cls):
        with patch.object(analytics, "_initialized", True):
            # Should not raise
            analytics.track("user-1", "test_event", {"key": "value"})

    mock_session.rollback.assert_called_once()
    mock_session.close.assert_called_once()


def test_track_skips_when_not_initialized():
    """analytics.track should silently skip when not initialized."""
    with patch.object(analytics, "_initialized", False):
        analytics.track("user-1", "test_event")  # Should not raise


def test_identify_delegates_to_track():
    """analytics.identify should call track with 'user_identified' event."""
    with patch.object(analytics, "track") as mock_track:
        analytics.identify("user-1", {"name": "Test"})
        mock_track.assert_called_once_with(
            "user-1", "user_identified", {"name": "Test"}
        )
