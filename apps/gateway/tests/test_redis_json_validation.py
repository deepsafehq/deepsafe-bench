"""Tests for JSON validation of Redis/DB data and result shaping helpers.

Covers:
- _shape_result_response from v1_router (COMPLETE, FAILED, PROCESSING, missing keys)
- _sanitize_result from routers.history (strips internal keys, handles empty dict)
"""

from routers.history import _sanitize_result
from routers.v1 import _shape_result_response

# ---------------------------------------------------------------------------
# _shape_result_response
# ---------------------------------------------------------------------------


class TestShapeResultResponse:
    """Unit tests for v1_router._shape_result_response."""

    def test_shape_result_response_handles_missing_keys(self):
        """Minimal dict with only status should not crash."""
        result = _shape_result_response("det_abc123", {"status": "COMPLETE"})
        assert result["id"] == "det_abc123"
        assert result["status"] == "complete"
        # verdict and confidence may be None but must be present
        assert "verdict" in result
        assert "confidence" in result

    def test_shape_result_response_complete(self):
        """COMPLETE status returns verdict, confidence, and media_type."""
        data = {
            "status": "COMPLETE",
            "created_at": "2026-04-10T12:00:00Z",
            "completed_at": "2026-04-10T12:00:05Z",
            "result": {
                "verdict": "fake",
                "confidence": 0.97,
                "media_type": "image",
            },
        }
        result = _shape_result_response("det_abc123", data)
        assert result["status"] == "complete"
        assert result["verdict"] == "fake"
        assert result["confidence"] == 0.97
        assert result["media_type"] == "image"
        assert result["created_at"] == "2026-04-10T12:00:00Z"
        assert result["completed_at"] == "2026-04-10T12:00:05Z"

    def test_shape_result_response_failed(self):
        """FAILED status includes error field."""
        data = {
            "status": "FAILED",
            "error": "Model timeout",
            "created_at": "2026-04-10T12:00:00Z",
        }
        result = _shape_result_response("det_abc123", data)
        assert result["status"] == "failed"
        assert result["error"] == "Model timeout"
        assert result["created_at"] == "2026-04-10T12:00:00Z"

    def test_shape_result_response_processing(self):
        """PROCESSING status returns a poll_url."""
        data = {
            "status": "PROCESSING",
            "created_at": "2026-04-10T12:00:00Z",
        }
        result = _shape_result_response("det_abc123", data)
        assert result["status"] == "processing"
        assert result["poll_url"] == "/v1/results/det_abc123"
        assert result["created_at"] == "2026-04-10T12:00:00Z"


# ---------------------------------------------------------------------------
# _sanitize_result
# ---------------------------------------------------------------------------


class TestSanitizeResult:
    """Unit tests for routers.history._sanitize_result."""

    def test_sanitize_result_strips_internal_keys(self):
        """Internal keys are removed from the result dict."""
        raw = {
            "verdict": "fake",
            "confidence": 0.95,
            "model_results": {"model_a": 0.9},
            "ensemble_method_used": "random_forest",
            "processing_mode": "gpu",
        }
        sanitized = _sanitize_result(raw)
        assert "model_results" not in sanitized
        assert "ensemble_method_used" not in sanitized
        assert "processing_mode" not in sanitized
        # Public keys survive
        assert sanitized["verdict"] == "fake"
        assert sanitized["confidence"] == 0.95

    def test_sanitize_result_handles_empty_dict(self):
        """Empty dict returns empty dict."""
        assert _sanitize_result({}) == {}
