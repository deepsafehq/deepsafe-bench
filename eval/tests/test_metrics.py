"""Unit tests for the metrics module."""

import numpy as np
import pytest


class TestEER:
    """Equal Error Rate computation."""

    def test_perfect_separation(self):
        """Near-perfect separation should yield EER close to 0."""
        from deepsafe_eval.metrics import compute_eer

        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        eer, threshold = compute_eer(labels, probs)
        assert eer < 0.01  # Near-perfect separation -> EER ~ 0

    def test_random_classifier(self):
        """A random classifier should yield EER close to 0.5."""
        from deepsafe_eval.metrics import compute_eer

        np.random.seed(42)
        labels = [0] * 500 + [1] * 500
        probs = list(np.random.rand(1000))
        eer, threshold = compute_eer(labels, probs)
        assert 0.35 < eer < 0.65  # Random -> EER ~ 0.5

    def test_returns_threshold(self):
        """EER threshold must be a valid probability value."""
        from deepsafe_eval.metrics import compute_eer

        labels = [0, 0, 1, 1]
        probs = [0.2, 0.4, 0.6, 0.8]
        eer, threshold = compute_eer(labels, probs)
        assert 0.0 <= threshold <= 1.0


class TestFPRatFNR:
    """FPR at fixed FNR."""

    def test_fpr_at_1pct_fnr(self):
        """FPR at 1% FNR should be a valid probability."""
        from deepsafe_eval.metrics import compute_fpr_at_fnr

        labels = [0] * 100 + [1] * 100
        probs = [i / 200 for i in range(200)]
        fpr = compute_fpr_at_fnr(labels, probs, target_fnr=0.01)
        assert 0.0 <= fpr <= 1.0

    def test_perfect_separation_zero_fpr(self):
        """Perfect separation should yield 0 FPR at any FNR."""
        from deepsafe_eval.metrics import compute_fpr_at_fnr

        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        fpr = compute_fpr_at_fnr(labels, probs, target_fnr=0.01)
        assert fpr == 0.0


class TestFNRatFPR:
    """FNR at fixed FPR."""

    def test_fnr_at_1pct_fpr(self):
        """FNR at 1% FPR should be a valid probability."""
        from deepsafe_eval.metrics import compute_fnr_at_fpr

        labels = [0] * 100 + [1] * 100
        probs = [i / 200 for i in range(200)]
        fnr = compute_fnr_at_fpr(labels, probs, target_fpr=0.01)
        assert 0.0 <= fnr <= 1.0

    def test_perfect_separation_zero_fnr(self):
        """Perfect separation should yield 0 FNR at any FPR."""
        from deepsafe_eval.metrics import compute_fnr_at_fpr

        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        fnr = compute_fnr_at_fpr(labels, probs, target_fpr=0.01)
        assert fnr == 0.0


class TestECE:
    """Expected Calibration Error."""

    def test_perfectly_calibrated(self):
        """Reasonably calibrated predictions should have low ECE."""
        from deepsafe_eval.metrics import compute_ece

        # Predicted prob roughly matches actual frequency
        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
        ece = compute_ece(labels, probs, n_bins=5)
        assert ece < 0.3  # Reasonably calibrated

    def test_completely_miscalibrated(self):
        """All predicted 0.9 but all actually 0 should have high ECE."""
        from deepsafe_eval.metrics import compute_ece

        labels = [0] * 100
        probs = [0.9] * 100
        ece = compute_ece(labels, probs, n_bins=10)
        assert ece > 0.8

    def test_returns_float_between_0_and_1(self):
        """ECE must be in [0, 1]."""
        from deepsafe_eval.metrics import compute_ece

        labels = [0, 1, 0, 1]
        probs = [0.3, 0.7, 0.4, 0.6]
        ece = compute_ece(labels, probs, n_bins=5)
        assert 0.0 <= ece <= 1.0


class TestBootstrapCI:
    """Bootstrap confidence intervals."""

    def test_ci_contains_point_estimate(self):
        """The CI must contain the point AUC estimate."""
        from deepsafe_eval.metrics import bootstrap_auc_ci

        labels = [0] * 50 + [1] * 50
        probs = [0.3] * 50 + [0.7] * 50
        point_auc, (lo, hi) = bootstrap_auc_ci(labels, probs, n_iter=100)
        assert lo <= point_auc <= hi

    def test_ci_width_decreases_with_samples(self):
        """Larger sample size should produce tighter CI."""
        from deepsafe_eval.metrics import bootstrap_auc_ci

        np.random.seed(42)
        labels_small = [0] * 20 + [1] * 20
        probs_small = list(np.random.rand(40))
        _, (lo_s, hi_s) = bootstrap_auc_ci(labels_small, probs_small, n_iter=100)

        labels_big = [0] * 200 + [1] * 200
        probs_big = list(np.random.rand(400))
        _, (lo_b, hi_b) = bootstrap_auc_ci(labels_big, probs_big, n_iter=100)

        assert (hi_b - lo_b) < (hi_s - lo_s)  # Bigger sample -> tighter CI


class TestPerGeneratorAUC:
    """Per-generator AUC with proper real-sample pooling."""

    def test_uses_global_real_pool(self):
        """Each generator's AUC should use the global real pool as negatives."""
        from deepsafe_eval.metrics import compute_per_generator_auc

        predictions = [
            {"label": 0, "generator": "coco", "ensemble_prob": 0.1},
            {"label": 0, "generator": "coco", "ensemble_prob": 0.2},
            {"label": 1, "generator": "dalle_3", "ensemble_prob": 0.8},
            {"label": 1, "generator": "dalle_3", "ensemble_prob": 0.9},
            {"label": 1, "generator": "flux_1", "ensemble_prob": 0.7},
        ]
        result = compute_per_generator_auc(predictions)
        assert "dalle_3" in result
        assert "flux_1" in result
        assert result["dalle_3"] > 0.0  # Must not be 0.0

    def test_skips_generators_with_one_sample(self):
        """Generators with only 1 fake sample should still compute AUC."""
        from deepsafe_eval.metrics import compute_per_generator_auc

        predictions = [
            {"label": 0, "generator": "coco", "ensemble_prob": 0.1},
            {"label": 1, "generator": "rare_gen", "ensemble_prob": 0.9},
        ]
        result = compute_per_generator_auc(predictions)
        # With only 1 fake sample, should still compute (using global reals)
        assert "rare_gen" in result


class TestComputeAllMetrics:
    """Integration: compute_all_metrics returns complete dict."""

    def test_returns_all_keys(self):
        """Must return all expected metric keys."""
        from deepsafe_eval.metrics import compute_all_metrics

        labels = [0] * 50 + [1] * 50
        probs = [0.3] * 50 + [0.7] * 50
        result = compute_all_metrics(labels, probs)
        assert "auc_roc" in result
        assert "eer" in result
        assert "ece" in result
        assert "fpr_at_1pct_fnr" in result
        assert "bootstrap_ci" in result
