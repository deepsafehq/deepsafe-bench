"""Tests for database models."""

from database import AnalysisHistory, CostTracking, ModelPerformance


def test_analysis_history_has_user_id_column():
    """AnalysisHistory should have user_id, not username."""
    columns = {c.name for c in AnalysisHistory.__table__.columns}
    assert "user_id" in columns
    assert "username" not in columns


def test_model_performance_has_required_columns():
    """ModelPerformance must have all spec-defined columns."""
    columns = {c.name for c in ModelPerformance.__table__.columns}
    expected = {
        "id",
        "analysis_id",
        "user_id",
        "model_name",
        "model_score",
        "raw_output",
        "inference_time_ms",
        "model_version",
        "status",
        "error_message",
        "media_type",
        "created_at",
    }
    assert expected.issubset(columns), f"Missing columns: {expected - columns}"


def test_model_performance_analysis_id_is_fk():
    """analysis_id must be a foreign key to analysis_history.id."""
    fks = {fk.target_fullname for fk in ModelPerformance.__table__.foreign_keys}
    assert "analysis_history.id" in fks


def test_cost_tracking_has_required_columns():
    """CostTracking must have all spec-defined columns."""
    columns = {c.name for c in CostTracking.__table__.columns}
    expected = {
        "id",
        "analysis_id",
        "user_id",
        "credits_consumed",
        "model_count",
        "total_inference_ms",
        "media_type",
        "file_size_bytes",
        "created_at",
    }
    assert expected.issubset(columns), f"Missing columns: {expected - columns}"
