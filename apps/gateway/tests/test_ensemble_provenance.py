"""Tests for the two-stage provenance boost in the ensemble."""

import pytest
from deepsafe_shared.ensemble import (
    PROVENANCE_SERVICES,
    _get_provenance_weight,
    apply_provenance_boost,
)


class TestProvenanceWeightTiers:
    def test_moderate_signal_weight(self):
        assert _get_provenance_weight(0.70) == 0.20

    def test_boundary_065_is_moderate(self):
        assert _get_provenance_weight(0.65) == 0.20

    def test_weak_signal_weight(self):
        assert _get_provenance_weight(0.55) == 0.05

    def test_boundary_064_is_weak(self):
        assert _get_provenance_weight(0.64) == 0.05

    def test_just_above_neutral(self):
        assert _get_provenance_weight(0.51) == 0.05


class TestProvenanceBoostFormula:
    def test_no_provenance_signals_returns_model_score(self):
        results = {
            "c2pa_checker": {"probability": 0.5},
            "sdxl_watermark_detector": {"probability": 0.5},
        }
        score, boosted = apply_provenance_boost(0.6, results)
        assert score == 0.6
        assert boosted is False

    def test_no_provenance_results_returns_model_score(self):
        results = {"npr": {"probability": 0.8}, "aide": {"probability": 0.7}}
        score, boosted = apply_provenance_boost(0.6, results)
        assert score == 0.6
        assert boosted is False

    def test_empty_results_returns_model_score(self):
        score, boosted = apply_provenance_boost(0.5, {})
        assert score == 0.5
        assert boosted is False

    def test_single_provenance_signal_boosts(self):
        results = {"c2pa_checker": {"probability": 0.95}}
        score, boosted = apply_provenance_boost(0.6, results)
        assert score > 0.6
        assert boosted is True

    def test_multiple_provenance_signals_uses_max(self):
        results = {
            "c2pa_checker": {"probability": 0.90},
            "audioseal_detector": {"probability": 0.95},
        }
        score_multi, _ = apply_provenance_boost(0.6, results)
        results_single = {"audioseal_detector": {"probability": 0.95}}
        score_single, _ = apply_provenance_boost(0.6, results_single)
        assert score_multi == score_single

    def test_boost_never_exceeds_1(self):
        results = {"c2pa_checker": {"probability": 0.99}}
        score, _ = apply_provenance_boost(0.99, results)
        assert score <= 1.0

    def test_boost_asymptotic(self):
        results = {"c2pa_checker": {"probability": 0.95}}
        score_low, _ = apply_provenance_boost(0.3, results)
        score_high, _ = apply_provenance_boost(0.9, results)
        assert (score_low - 0.3) > (score_high - 0.9)

    def test_provenance_error_results_ignored(self):
        results = {
            "c2pa_checker": {"error": "connection refused"},
            "sdxl_watermark_detector": {"probability": 0.70},
        }
        score, boosted = apply_provenance_boost(0.6, results)
        assert score > 0.6
        assert boosted is True

    def test_trustmark_in_provenance_services(self):
        assert "trustmark_detector" in PROVENANCE_SERVICES

    def test_known_service_names(self):
        expected = {
            "c2pa_checker",
            "sdxl_watermark_detector",
            "audioseal_detector",
            "videoseal_detector",
            "trustmark_detector",
        }
        assert expected == PROVENANCE_SERVICES


class TestProvenanceBoostMath:
    """Verify exact math with tiered weights."""

    def test_moderate_signal_borderline_model(self):
        """SDXL watermark (0.70) on model score 0.50."""
        results = {"sdxl_watermark_detector": {"probability": 0.70}}
        score, _ = apply_provenance_boost(0.5, results)
        # weight=0.20 for >=0.65
        expected = 0.5 + (0.70 - 0.5) * 0.20 * (1.0 - 0.5)
        assert abs(score - expected) < 1e-10

    def test_moderate_signal_strong_model(self):
        """SDXL watermark (0.70) on model score 0.70."""
        results = {"sdxl_watermark_detector": {"probability": 0.70}}
        score, _ = apply_provenance_boost(0.7, results)
        expected = 0.7 + (0.70 - 0.5) * 0.20 * (1.0 - 0.7)
        assert abs(score - expected) < 1e-10

    def test_weak_signal_neutral_model(self):
        """Weak signal (0.55) on model score 0.50."""
        results = {"c2pa_checker": {"probability": 0.55}}
        score, _ = apply_provenance_boost(0.5, results)
        # weight=0.05 for <0.65
        expected = 0.5 + (0.55 - 0.5) * 0.05 * (1.0 - 0.5)
        assert abs(score - expected) < 1e-10

    def test_trustmark_signal_boost(self):
        """TrustMark (0.65) on model score 0.60."""
        results = {"trustmark_detector": {"probability": 0.65}}
        score, _ = apply_provenance_boost(0.6, results)
        # weight=0.20 for >=0.65
        expected = 0.6 + (0.65 - 0.5) * 0.20 * (1.0 - 0.6)
        assert abs(score - expected) < 1e-10
