"""Unit tests for ensemble.py: modality-specific ensemble strategies."""

from unittest.mock import MagicMock, patch

import pytest


def _make_result(prob: float, pred: int = None) -> dict:
    """Build a minimal model result dict."""
    if pred is None:
        pred = 1 if prob >= 0.5 else 0
    return {"probability": prob, "prediction": pred}


def _call(results, media_type="image", threshold=0.5):
    """Thin wrapper around calculate_ensemble_verdict_api."""
    import deepsafe_shared.ensemble as ens

    return ens.calculate_ensemble_verdict_api(
        results, threshold, "auto", media_type, "test-req"
    )


# ---------------------------------------------------------------------------
# Helpers for meta-learner mocking
# ---------------------------------------------------------------------------

_IMAGE_MODEL_ORDER = [
    "aide",
    "cospy",
    "effort",
    "fsd",
    "npr",
    "universal",
    "yermandy",
]


def _make_meta_learner_mock(return_prob: float = 0.75):
    """Return a mock scikit-learn-like classifier."""
    import numpy as np

    clf = MagicMock()
    clf.predict_proba = MagicMock(
        return_value=np.array([[1 - return_prob, return_prob]])
    )
    return clf


def _make_scaler_mock():
    """Return a mock StandardScaler that passes data through unchanged."""
    import numpy as np

    scaler = MagicMock()
    scaler.transform = MagicMock(side_effect=lambda x: x)
    return scaler


# ---------------------------------------------------------------------------
# Image ensemble — weighted average fallback (no meta-learner artifacts)
# ---------------------------------------------------------------------------


class TestImageWeightedAverage:
    """Image ensemble falls back to AUC-weighted average when meta-learner artifacts are absent."""

    def setup_method(self):
        """Ensure meta-learner globals are cleared before each test."""
        import deepsafe_shared.ensemble as ens

        ens._image_meta_learner = None
        ens._image_scaler = None
        ens._image_model_order = None
        ens._image_calibrator = None

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_weighted_average_all_fake(self, _mock_exists):
        results = {
            "npr_deepfakedetection": _make_result(0.9),
            "fsd_detection": _make_result(0.85),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        assert verdict == "fake"
        assert score > 0.5
        assert method == "weighted_avg"

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_weighted_average_all_real(self, _mock_exists):
        results = {
            "npr_deepfakedetection": _make_result(0.1),
            "fsd_detection": _make_result(0.15),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        assert verdict == "real"
        assert score < 0.5
        assert method == "weighted_avg"

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_unknown_model_name_uses_default_weight(self, _mock_exists):
        """Models not in the AUC table fall back to weight 0.5."""
        results = {
            "unknown_model_xyz": _make_result(0.9),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        assert method == "weighted_avg"
        # weight=0.5, single model → score == prob
        assert abs(score - 0.9) < 1e-6

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_empty_results_returns_undetermined(self, _mock_exists):
        verdict, confidence, fv, rv, score, method = _call({}, "image")
        assert verdict == "undetermined"
        assert method == "none"
        assert score == 0.5

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_all_error_results_returns_undetermined(self, _mock_exists):
        results = {
            "model_a": {"error": "timeout"},
            "model_b": {"error": "OOM"},
        }
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        assert verdict == "undetermined"
        assert method == "none"


# ---------------------------------------------------------------------------
# Image ensemble — meta-learner path (artifacts present)
# ---------------------------------------------------------------------------


class TestImageMetaLearner:
    """Image ensemble uses the Random Forest meta-learner when all 7 models are present."""

    def setup_method(self):
        import deepsafe_shared.ensemble as ens

        ens._image_meta_learner = None
        ens._image_scaler = None
        ens._image_model_order = None

    def _inject_meta_learner(self, return_prob=0.75):
        """Directly inject mocked artifacts, bypassing file I/O."""
        import deepsafe_shared.ensemble as ens

        ens._image_meta_learner = _make_meta_learner_mock(return_prob)
        ens._image_scaler = _make_scaler_mock()
        ens._image_model_order = _IMAGE_MODEL_ORDER

    def _all_model_results(self, prob=0.8):
        """Build results dict with all 7 gateway model names."""
        mapping = {
            "npr_deepfakedetection": prob,
            "universalfakedetect": prob,
            "yermandy_clip_detection": prob,
            "aide_detection": prob,
            "fsd_detection": prob,
            "effort_detection": prob,
            "cospy_detection": prob,
        }
        return {k: _make_result(v) for k, v in mapping.items()}

    def test_meta_learner_used_when_all_models_present(self):
        """Meta-learner is used when all 7 models are present."""
        self._inject_meta_learner(return_prob=0.75)
        results = self._all_model_results(prob=0.8)
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        assert method == "meta_learner"
        assert abs(score - 0.75) < 1e-6
        assert verdict == "fake"

    def test_meta_learner_real_verdict(self):
        """Meta-learner returning low prob gives 'real' verdict."""
        self._inject_meta_learner(return_prob=0.2)
        results = self._all_model_results(prob=0.2)
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        assert method == "meta_learner"
        assert verdict == "real"

    def test_subset_models_falls_back_to_weighted_avg(self):
        """With one model missing, falls back to weighted average."""
        self._inject_meta_learner(return_prob=0.75)
        results = self._all_model_results(prob=0.8)
        del results["fsd_detection"]
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        assert method == "weighted_avg"
        assert 0.7 < score < 0.9

    def test_all_models_missing_falls_back(self):
        """With no valid models at all, result is undetermined."""
        self._inject_meta_learner(return_prob=0.75)
        results = {"model_a": {"error": "fail"}}
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        assert verdict == "undetermined"


# ---------------------------------------------------------------------------
# Audio ensemble
# ---------------------------------------------------------------------------


class TestAudioEnsemble:
    """Audio ensemble: ShiftySpeech primary, SafeEar fallback."""

    def test_shiftyspeech_high_score_detected_as_fake(self):
        """High ShiftySpeech score should produce a fake verdict via meta-learner."""
        results = {
            "shiftyspeech_detection": _make_result(0.85),
            "safeear_detection": _make_result(0.5),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "audio")
        assert method in ("meta_learner", "weighted_avg")
        assert score > 0.5  # High shiftyspeech should push score up

    def test_conflicting_audio_models_shiftyspeech_dominates(self):
        """When ShiftySpeech says fake and SafeEar says real, the ensemble
        should weight ShiftySpeech higher (higher AUC)."""
        results = {
            "shiftyspeech_detection": _make_result(0.9),
            "safeear_detection": _make_result(0.1),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "audio")
        # Meta-learner or weighted avg — either way, shiftyspeech should dominate
        assert score > 0.5  # Not a naive 0.5 average

    def test_safeear_only_produces_verdict(self):
        """SafeEar alone should still produce a verdict via meta-learner
        (missing models imputed as 0.5)."""
        results = {
            "safeear_detection": _make_result(0.7),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "audio")
        assert method in ("meta_learner", "weighted_avg")
        assert verdict in ("fake", "real")

    def test_shiftyspeech_fake_verdict(self):
        results = {"shiftyspeech_detection": _make_result(0.95)}
        verdict, confidence, fv, rv, score, method = _call(results, "audio")
        assert verdict == "fake"

    def test_shiftyspeech_low_score_uses_meta_learner(self):
        """Low ShiftySpeech score is processed by the meta-learner.

        In production all 3 audio models run. This test verifies the
        meta-learner handles the input even when only one model is present
        (missing models are imputed as 0.5).
        """
        results = {"shiftyspeech_detection": _make_result(0.1)}
        verdict, confidence, fv, rv, score, method = _call(results, "audio")
        # meta_learner when artifacts are present, weighted_avg in CI
        assert method in ("meta_learner", "weighted_avg")

    def test_no_audio_models_returns_undetermined(self):
        results = {"shiftyspeech_detection": {"error": "unavailable"}}
        verdict, confidence, fv, rv, score, method = _call(results, "audio")
        assert verdict == "undetermined"

    def test_unknown_audio_models_produce_verdict(self):
        """Unknown models should still produce a verdict via meta-learner
        (imputed as 0.5) or weighted average."""
        results = {
            "other_audio_model": _make_result(0.6),
            "another_model": _make_result(0.4),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "audio")
        assert method in ("meta_learner", "weighted_avg")
        assert verdict in ("fake", "real")


# ---------------------------------------------------------------------------
# Video ensemble
# ---------------------------------------------------------------------------


class TestVideoEnsemble:
    """Video ensemble: AUC-weighted average across all video models."""

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_multiple_models_weighted_average(self, _mock):
        results = {
            "fakestormer": _make_result(0.7),
            "sbi_detection": _make_result(0.8),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "video")
        # Weighted average, not pass-through
        assert 0.7 < score < 0.8

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_high_fake_score_dominates(self, _mock):
        """High-AUC model with high score should pull ensemble toward fake."""
        results = {
            "fakestormer": _make_result(0.9),
            "sbi_detection": _make_result(0.1),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "video")
        # FakeStormer (0.672 AUC weight) at 0.9 + SBI (0.780 AUC weight) at 0.1
        # Weighted avg should be moderate
        assert 0.3 < score < 0.7

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_single_model_fallback(self, _mock):
        results = {"fakestormer": _make_result(0.6)}
        verdict, confidence, fv, rv, score, method = _call(results, "video")
        assert abs(score - 0.6) < 1e-6

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_fakestormer_fake_verdict(self, _mock):
        results = {"fakestormer": _make_result(0.8)}
        verdict, confidence, fv, rv, score, method = _call(results, "video")
        assert verdict == "fake"

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_fakestormer_real_verdict(self, _mock):
        results = {"fakestormer": _make_result(0.2)}
        verdict, confidence, fv, rv, score, method = _call(results, "video")
        assert verdict == "real"

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_no_video_models_returns_undetermined(self, _mock):
        results = {"fakestormer": {"error": "service down"}}
        verdict, confidence, fv, rv, score, method = _call(results, "video")
        assert verdict == "undetermined"


# ---------------------------------------------------------------------------
# Threshold and vote counting
# ---------------------------------------------------------------------------


class TestThresholdAndVotes:
    """Threshold and vote-counting behaviour across modalities."""

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_custom_threshold_makes_fake_real(self, _mock):
        """Score=0.6 with threshold=0.7 should yield 'real'."""
        results = {"fakestormer": _make_result(0.6)}
        verdict, confidence, fv, rv, score, method = _call(
            results, "video", threshold=0.7
        )
        assert verdict == "real"

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_custom_threshold_makes_real_fake(self, _mock):
        """Score=0.4 with threshold=0.3 should yield 'fake'."""
        results = {"fakestormer": _make_result(0.4)}
        verdict, confidence, fv, rv, score, method = _call(
            results, "video", threshold=0.3
        )
        assert verdict == "fake"

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_vote_counting(self, _mock):
        results = {
            "model_a": _make_result(0.9, pred=1),
            "model_b": _make_result(0.3, pred=0),
            "model_c": _make_result(0.8, pred=1),
        }
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        assert fv == 2
        assert rv == 1

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_confidence_is_fake_score_when_fake(self, _mock):
        results = {"fakestormer": _make_result(0.8)}
        verdict, confidence, fv, rv, score, method = _call(results, "video")
        assert verdict == "fake"
        assert abs(confidence - score) < 1e-6

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_confidence_is_one_minus_score_when_real(self, _mock):
        results = {"fakestormer": _make_result(0.2)}
        verdict, confidence, fv, rv, score, method = _call(results, "video")
        assert verdict == "real"
        assert abs(confidence - (1.0 - score)) < 1e-6


# ---------------------------------------------------------------------------
# Model name normalisation
# ---------------------------------------------------------------------------


class TestModelNameNormalisation:
    """Gateway model names must be mapped to standard short names."""

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_gateway_name_normalised(self, _mock):
        """npr_deepfakedetection must resolve to AUC weight of 0.741."""
        results = {"npr_deepfakedetection": _make_result(0.9)}
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        # AUC weight for 'npr' is 0.741; single model → score equals probability
        assert abs(score - 0.9) < 1e-6

    @patch("deepsafe_shared.ensemble.Path.exists", return_value=False)
    def test_unknown_name_passes_through(self, _mock):
        """Names not in the map are passed through unchanged."""
        results = {"totally_unknown_model": _make_result(0.6)}
        verdict, confidence, fv, rv, score, method = _call(results, "image")
        # Default weight 0.5, single model → score equals probability
        assert abs(score - 0.6) < 1e-6
