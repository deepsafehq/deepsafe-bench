"""Tests for Tier 1 adaptation and the generalization check."""

import pytest

from deepsafe.fit import GeneralizationReport, LogisticMetaLearner, fit_ensemble


class TestLogisticMetaLearner:
    def test_learns_a_separable_problem(self):
        features = [[0.0], [0.1], [0.9], [1.0]]
        labels = [0, 0, 1, 1]
        model = LogisticMetaLearner()
        model.fit(features, labels)
        assert model.predict_proba([0.05]) < 0.5
        assert model.predict_proba([0.95]) > 0.5

    def test_probabilities_stay_in_range(self):
        model = LogisticMetaLearner()
        model.fit([[0.0], [1.0]], [0, 1])
        for x in (-1e6, 0.0, 1e6):
            assert 0.0 <= model.predict_proba([x]) <= 1.0

    def test_empty_training_set_raises(self):
        with pytest.raises(ValueError, match="no training rows"):
            LogisticMetaLearner().fit([], [])


def _rows(n_generators=8, per_generator=10):
    """Synthetic rows where the score is informative about the label."""
    rows = []
    for g in range(n_generators):
        for i in range(per_generator):
            is_fake = i % 2 == 0
            rows.append({
                "label": "fake" if is_fake else "real",
                "generator": f"gen{g}",
                "prob_a": 0.8 if is_fake else 0.2,
                "prob_b": 0.7 if is_fake else 0.3,
            })
    return rows


class TestFitEnsemble:
    def test_splits_by_generator_not_by_sample(self):
        """Held-out generators must not appear in training."""
        _, report = fit_ensemble(_rows(), ["prob_a", "prob_b"])
        assert report.held_out_generators
        assert len(report.held_out_generators) < 8

    def test_refuses_too_few_generators(self):
        rows = _rows(n_generators=2)
        with pytest.raises(ValueError, match="at least 4 distinct generators"):
            fit_ensemble(rows, ["prob_a"])

    def test_refuses_when_no_usable_rows(self):
        rows = [{"label": "fake", "generator": "g", "prob_a": "n/a"}]
        with pytest.raises(ValueError, match="no rows"):
            fit_ensemble(rows, ["prob_a"])

    def test_is_deterministic_for_a_seed(self):
        a = fit_ensemble(_rows(), ["prob_a"], seed=7)[1]
        b = fit_ensemble(_rows(), ["prob_a"], seed=7)[1]
        assert a.held_out_generators == b.held_out_generators


class TestGeneralizationReport:
    def test_flags_a_degraded_fine_tune(self):
        report = GeneralizationReport(0.99, 0.60, 0.80, ["sora"])
        assert report.transferred is False
        assert "WARNING" in report.render()
        assert "overfitting" in report.render()

    def test_accepts_an_improvement(self):
        report = GeneralizationReport(0.95, 0.85, 0.80, ["sora"])
        assert report.transferred is True
        assert "WARNING" not in report.render()

    def test_small_regression_is_tolerated(self):
        """Noise below one AUC point must not trigger a false alarm."""
        assert GeneralizationReport(0.9, 0.795, 0.80, ["g"]).transferred is True

    def test_missing_baseline_does_not_claim_failure(self):
        assert GeneralizationReport(0.9, None, None, []).transferred is True
