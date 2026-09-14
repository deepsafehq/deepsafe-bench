"""Tests for ModelPerformance insertion in run_detection."""

from unittest.mock import MagicMock

from database import ModelPerformance


def test_model_performance_rows_created_per_model():
    """One ModelPerformance row per model per analysis."""
    mock_db = MagicMock()
    model_results = {
        "fsd": {"probability": 0.85, "prediction": 1},
        "effort": {"probability": 0.72, "prediction": 1},
        "safeear": {"error": "timeout"},
    }

    from main import _insert_model_performance

    _insert_model_performance(mock_db, 1, "user-123", model_results, "image")

    assert mock_db.add.call_count == 3

    first_call = mock_db.add.call_args_list[0]
    row = first_call[0][0]
    assert isinstance(row, ModelPerformance)
    assert row.model_name == "fsd"
    assert row.model_score == 0.85
    assert row.status == "success"

    third_call = mock_db.add.call_args_list[2]
    error_row = third_call[0][0]
    assert error_row.model_name == "safeear"
    assert error_row.status == "error"
    assert error_row.error_message == "timeout"
