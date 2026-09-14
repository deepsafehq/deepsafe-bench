"""Tests for CostTracking insertion in run_detection."""

from unittest.mock import MagicMock

from database import CostTracking


def test_cost_tracking_row_created():
    """One CostTracking row per analysis."""
    mock_db = MagicMock()

    from main import _insert_cost_tracking

    _insert_cost_tracking(
        mock_db,
        analysis_id=1,
        user_id="user-123",
        model_count=3,
        total_inference_ms=1500.0,
        media_type="image",
        file_size_bytes=1024000,
    )

    mock_db.add.assert_called_once()
    row = mock_db.add.call_args[0][0]
    assert isinstance(row, CostTracking)
    assert row.credits_consumed == 1
    assert row.model_count == 3
