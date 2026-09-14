"""Tests for Celery periodic tasks."""

from unittest.mock import MagicMock, patch


def test_refresh_usage_summary_executes_sql():
    """refresh_usage_summary should call REFRESH MATERIALIZED VIEW."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    with patch("tasks.engine", mock_engine):
        from tasks import refresh_usage_summary

        refresh_usage_summary()

    mock_conn.execute.assert_called_once()
    sql_text = str(mock_conn.execute.call_args[0][0])
    assert "REFRESH MATERIALIZED VIEW" in sql_text
