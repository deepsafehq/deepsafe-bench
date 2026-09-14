"""Tests for the provenance override in the detection pipeline.

Verifies that provenance services (e.g. C2PA, watermark detectors) can
override the ensemble verdict when cryptographic proof of AI generation
is found (probability >= PROVENANCE_OVERRIDE_THRESHOLD).

With single-phase parallel dispatch, all services (models + provenance)
run concurrently.  The override is evaluated *after* the fan-out completes.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy dependencies that services.detection imports at module level.
# These are not available in the test environment.
_mock_deps = MagicMock()
_mock_deps.redis_client = MagicMock()
_mock_deps.minio_client = MagicMock()
_mock_deps.MINIO_JOBS_BUCKET = "jobs"
sys.modules.setdefault("dependencies", _mock_deps)

# The threshold constant under test.
from deepsafe_shared.ensemble import PROVENANCE_OVERRIDE_THRESHOLD, PROVENANCE_SERVICES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(media_type, endpoints):
    """Build a minimal ALL_MODEL_CONFIGS dict for testing."""
    return {
        "default_threshold": 0.5,
        "default_ensemble_method": "average",
        "media_types": {
            media_type: {
                "model_endpoints": endpoints,
            }
        },
    }


def _mock_db():
    """Return a lightweight mock SQLAlchemy session."""
    db = MagicMock()
    db.add = MagicMock()
    db.flush = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    # Give the flushed record an id attribute.
    db.add.side_effect = lambda obj: setattr(obj, "id", 42)
    return db


# ---------------------------------------------------------------------------
# Tests for threshold constant
# ---------------------------------------------------------------------------


class TestProvenanceOverrideThreshold:
    def test_threshold_value(self):
        assert PROVENANCE_OVERRIDE_THRESHOLD == 0.80

    def test_threshold_exported_from_ensemble(self):
        import deepsafe_shared.ensemble as ensemble

        assert hasattr(ensemble, "PROVENANCE_OVERRIDE_THRESHOLD")


# ---------------------------------------------------------------------------
# Tests for _run_sync_detection override path
# ---------------------------------------------------------------------------


class TestSyncDetectionOverride:
    """Tests for provenance override in _run_sync_detection."""

    @patch("services.detection._insert_cost_tracking")
    @patch("services.detection._insert_model_performance")
    @patch("services.detection.query_model_api")
    def test_c2pa_095_triggers_override(self, mock_query, mock_perf, mock_cost):
        """C2PA score of 0.95 should trigger override and return fake."""
        endpoints = {
            "c2pa_checker": "http://c2pa:9001/predict",
            "npr_deepfakedetection": "http://npr:8001/predict",
        }
        config = _make_config("image", endpoints)

        mock_query.side_effect = lambda name, *a, **k: (
            {"probability": 0.95, "prediction": 1}
            if name == "c2pa_checker"
            else {"probability": 0.3, "prediction": 0}
        )

        db = _mock_db()

        with patch("services.detection.ALL_MODEL_CONFIGS", config):
            from services.detection import _run_sync_detection

            result = _run_sync_detection(
                file_bytes=b"fake-image-data",
                content_type="image/jpeg",
                media_type="image",
                user_id="user-123",
                detection_id="det_override1",
                db=db,
            )

        assert result["verdict"] == "fake"
        assert result["confidence"] == 0.95
        assert result["media_type"] == "image"

        # With single-phase dispatch, all services are called in parallel.
        # The override verdict is still correct despite models being called.
        call_names = [c.args[0] for c in mock_query.call_args_list]
        assert "c2pa_checker" in call_names
        assert "npr_deepfakedetection" in call_names

    @patch("services.detection._insert_cost_tracking")
    @patch("services.detection._insert_model_performance")
    @patch("services.detection.query_model_api")
    @patch("services.detection.calculate_ensemble_verdict_api")
    def test_c2pa_060_no_override(
        self, mock_ensemble, mock_query, mock_perf, mock_cost
    ):
        """C2PA score of 0.60 should NOT trigger override."""
        endpoints = {
            "c2pa_checker": "http://c2pa:9001/predict",
            "npr_deepfakedetection": "http://npr:8001/predict",
        }
        config = _make_config("image", endpoints)

        mock_query.side_effect = lambda name, *a, **k: (
            {"probability": 0.60, "prediction": 1}
            if name == "c2pa_checker"
            else {"probability": 0.7, "prediction": 1}
        )
        mock_ensemble.return_value = ("fake", 0.7, 1, 0, 0.7, "weighted_avg")

        db = _mock_db()

        with patch("services.detection.ALL_MODEL_CONFIGS", config):
            from services.detection import _run_sync_detection

            result = _run_sync_detection(
                file_bytes=b"fake-image-data",
                content_type="image/jpeg",
                media_type="image",
                user_id="user-123",
                detection_id="det_no_override",
                db=db,
            )

        # Models should have been called (Phase 1 ran).
        call_names = [c.args[0] for c in mock_query.call_args_list]
        assert "npr_deepfakedetection" in call_names
        # Ensemble was invoked.
        mock_ensemble.assert_called_once()

    @patch("services.detection._insert_cost_tracking")
    @patch("services.detection._insert_model_performance")
    @patch("services.detection.query_model_api")
    def test_watermark_092_triggers_override(self, mock_query, mock_perf, mock_cost):
        """Watermark score of 0.92 is above 0.80 -- triggers override."""
        endpoints = {
            "sdxl_watermark_detector": "http://sdxl:9002/predict",
            "npr_deepfakedetection": "http://npr:8001/predict",
        }
        config = _make_config("image", endpoints)

        mock_query.side_effect = lambda name, *a, **k: (
            {"probability": 0.92, "prediction": 1}
            if name == "sdxl_watermark_detector"
            else {"probability": 0.4, "prediction": 0}
        )

        db = _mock_db()

        with patch("services.detection.ALL_MODEL_CONFIGS", config):
            from services.detection import _run_sync_detection

            result = _run_sync_detection(
                file_bytes=b"fake-image-data",
                content_type="image/jpeg",
                media_type="image",
                user_id="user-123",
                detection_id="det_watermark_override",
                db=db,
            )

        assert result["verdict"] == "fake"
        assert result["confidence"] == 0.92
        # With single-phase dispatch, all services are called in parallel.
        call_names = [c.args[0] for c in mock_query.call_args_list]
        assert "sdxl_watermark_detector" in call_names
        assert "npr_deepfakedetection" in call_names

    @patch("services.detection._insert_cost_tracking")
    @patch("services.detection._insert_model_performance")
    @patch("services.detection.query_model_api")
    @patch("services.detection.calculate_ensemble_verdict_api")
    def test_all_provenance_neutral_no_override(
        self, mock_ensemble, mock_query, mock_perf, mock_cost
    ):
        """All provenance at 0.5 (neutral) should not trigger override."""
        endpoints = {
            "c2pa_checker": "http://c2pa:9001/predict",
            "sdxl_watermark_detector": "http://sdxl:9002/predict",
            "npr_deepfakedetection": "http://npr:8001/predict",
        }
        config = _make_config("image", endpoints)

        mock_query.side_effect = lambda name, *a, **k: (
            {"probability": 0.5, "prediction": 0}
            if name in PROVENANCE_SERVICES
            else {"probability": 0.3, "prediction": 0}
        )
        mock_ensemble.return_value = ("real", 0.7, 0, 1, 0.3, "weighted_avg")

        db = _mock_db()

        with patch("services.detection.ALL_MODEL_CONFIGS", config):
            from services.detection import _run_sync_detection

            result = _run_sync_detection(
                file_bytes=b"real-image-data",
                content_type="image/jpeg",
                media_type="image",
                user_id="user-123",
                detection_id="det_neutral",
                db=db,
            )

        mock_ensemble.assert_called_once()

    @patch("services.detection._insert_cost_tracking")
    @patch("services.detection._insert_model_performance")
    @patch("services.detection.query_model_api")
    def test_override_works_with_some_provenance_errors(
        self, mock_query, mock_perf, mock_cost
    ):
        """Override triggers even if some provenance services error."""
        endpoints = {
            "c2pa_checker": "http://c2pa:9001/predict",
            "sdxl_watermark_detector": "http://sdxl:9002/predict",
            "npr_deepfakedetection": "http://npr:8001/predict",
        }
        config = _make_config("image", endpoints)

        def _side_effect(name, *a, **k):
            if name == "c2pa_checker":
                return {"probability": 0.97, "prediction": 1}
            if name == "sdxl_watermark_detector":
                return {"error": "connection refused"}
            return {"probability": 0.4, "prediction": 0}

        mock_query.side_effect = _side_effect

        db = _mock_db()

        with patch("services.detection.ALL_MODEL_CONFIGS", config):
            from services.detection import _run_sync_detection

            result = _run_sync_detection(
                file_bytes=b"fake-image-data",
                content_type="image/jpeg",
                media_type="image",
                user_id="user-123",
                detection_id="det_partial_err",
                db=db,
            )

        assert result["verdict"] == "fake"
        assert result["confidence"] == 0.97
        # With single-phase dispatch, all services are called in parallel.
        call_names = [c.args[0] for c in mock_query.call_args_list]
        assert "npr_deepfakedetection" in call_names

    @patch("services.detection._insert_cost_tracking")
    @patch("services.detection._insert_model_performance")
    @patch("services.detection.query_model_api")
    def test_override_method_used_is_provenance_override(
        self, mock_query, mock_perf, mock_cost
    ):
        """When override triggers, the DB record uses provenance_override."""
        endpoints = {
            "c2pa_checker": "http://c2pa:9001/predict",
            "npr_deepfakedetection": "http://npr:8001/predict",
        }
        config = _make_config("image", endpoints)

        mock_query.side_effect = lambda name, *a, **k: (
            {"probability": 0.98, "prediction": 1}
            if name == "c2pa_checker"
            else {"probability": 0.3, "prediction": 0}
        )

        db = _mock_db()

        with patch("services.detection.ALL_MODEL_CONFIGS", config):
            from services.detection import _run_sync_detection

            result = _run_sync_detection(
                file_bytes=b"fake-image-data",
                content_type="image/jpeg",
                media_type="image",
                user_id="user-123",
                detection_id="det_method_check",
                db=db,
            )

        # Check the AnalysisHistory record written to the DB.
        add_calls = db.add.call_args_list
        history_obj = None
        for call in add_calls:
            obj = call.args[0]
            if hasattr(obj, "ensemble_method"):
                history_obj = obj
                break

        assert history_obj is not None
        assert history_obj.ensemble_method == "provenance_override"

    @patch("services.detection._insert_cost_tracking")
    @patch("services.detection._insert_model_performance")
    @patch("services.detection.query_model_api")
    @patch("services.detection.calculate_ensemble_verdict_api")
    def test_no_provenance_endpoints_skips_phase0(
        self, mock_ensemble, mock_query, mock_perf, mock_cost
    ):
        """When no provenance services are configured, go straight to models."""
        endpoints = {
            "npr_deepfakedetection": "http://npr:8001/predict",
        }
        config = _make_config("image", endpoints)

        mock_query.return_value = {"probability": 0.7, "prediction": 1}
        mock_ensemble.return_value = ("fake", 0.7, 1, 0, 0.7, "weighted_avg")

        db = _mock_db()

        with patch("services.detection.ALL_MODEL_CONFIGS", config):
            from services.detection import _run_sync_detection

            result = _run_sync_detection(
                file_bytes=b"fake-image-data",
                content_type="image/jpeg",
                media_type="image",
                user_id="user-123",
                detection_id="det_no_prov",
                db=db,
            )

        assert result["verdict"] == "fake"
        mock_ensemble.assert_called_once()
