"""Tests for TrustMark watermark provenance detector."""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "apps/inference"))


def _make_jpeg_bytes(width=100, height=100, color="white"):
    """Create minimal JPEG bytes for testing."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG")
    return buf.getvalue()


class TestTrustMarkScoreMapping:
    """Test score mapping without loading real TrustMark model."""

    def test_detected_returns_065(self):
        from models.trustmark_wm import _map_detection_to_probability

        assert _map_detection_to_probability(True) == 0.65

    def test_not_detected_returns_050(self):
        from models.trustmark_wm import _map_detection_to_probability

        assert _map_detection_to_probability(False) == 0.50


class TestTrustMarkPredictor:
    """Test predictor wrapper logic with mocked TrustMark model."""

    def _make_predictor(self, decode_return):
        """Create a predictor with a mocked TrustMark instance."""
        from models.trustmark_wm import TrustMarkPredictor

        predictor = TrustMarkPredictor()
        predictor._device = MagicMock()
        predictor._loaded = True
        mock_tm = MagicMock()
        mock_tm.decode.return_value = decode_return
        predictor._tm = mock_tm
        return predictor

    def test_predict_watermarked_image(self):
        predictor = self._make_predictor(("0110101", True, 1))
        result = predictor._run(_make_jpeg_bytes())
        assert result["probability"] == 0.65
        assert result["prediction"] == "fake"

    def test_predict_clean_image(self):
        predictor = self._make_predictor(("", False, -1))
        result = predictor._run(_make_jpeg_bytes())
        assert result["probability"] == 0.50
        assert result["prediction"] == "real"

    def test_predict_invalid_bytes_returns_neutral(self):
        predictor = self._make_predictor(None)
        predictor._tm.decode.side_effect = Exception("bad image")
        result = predictor._run(b"not an image")
        assert result["probability"] == 0.50

    def test_predict_empty_bytes_returns_neutral(self):
        predictor = self._make_predictor(None)
        result = predictor._run(b"")
        assert result["probability"] == 0.50

    def test_result_contains_required_keys(self):
        predictor = self._make_predictor(("010101", True, 0))
        result = predictor._run(_make_jpeg_bytes())
        assert "probability" in result
        assert "prediction" in result

    def test_decode_called_with_pil_image(self):
        predictor = self._make_predictor(("", False, -1))
        predictor._run(_make_jpeg_bytes())
        call_args = predictor._tm.decode.call_args
        assert isinstance(call_args[0][0], Image.Image)
        assert call_args[1]["MODE"] == "binary"
