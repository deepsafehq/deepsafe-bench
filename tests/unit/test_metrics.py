"""Tests for the dependency-free scoring primitives."""

import pytest

from deepsafe import metrics


class TestRocAuc:
    def test_perfect_separation_is_one(self):
        pairs = [(0.1, False), (0.2, False), (0.8, True), (0.9, True)]
        assert metrics.roc_auc(pairs) == 1.0

    def test_inverted_separation_is_zero(self):
        pairs = [(0.9, False), (0.8, False), (0.2, True), (0.1, True)]
        assert metrics.roc_auc(pairs) == 0.0

    def test_all_ties_is_one_half(self):
        """A detector outputting a constant must score exactly chance."""
        pairs = [(0.5, True), (0.5, False), (0.5, True), (0.5, False)]
        assert metrics.roc_auc(pairs) == 0.5

    def test_matches_known_value(self):
        # Positives 0.4, 0.8 against negatives 0.1, 0.5. Of the four
        # positive/negative comparisons, three favour the positive: 3/4.
        pairs = [(0.1, False), (0.5, False), (0.4, True), (0.8, True)]
        assert metrics.roc_auc(pairs) == pytest.approx(0.75)

    def test_returns_none_when_one_class_missing(self):
        assert metrics.roc_auc([(0.5, True), (0.9, True)]) is None
        assert metrics.roc_auc([]) is None

    def test_ignores_none_scores(self):
        pairs = [(None, True), (0.9, True), (0.1, False)]
        assert metrics.roc_auc(pairs) == 1.0


class TestRecallAndFpr:
    def test_recall_counts_positives_above_threshold(self):
        pairs = [(0.9, True), (0.4, True), (0.8, False)]
        assert metrics.recall_at(pairs, 0.5) == pytest.approx(0.5)

    def test_fpr_counts_negatives_above_threshold(self):
        pairs = [(0.9, False), (0.1, False), (0.95, True)]
        assert metrics.false_positive_rate(pairs, 0.5) == pytest.approx(0.5)

    def test_threshold_is_inclusive(self):
        assert metrics.recall_at([(0.5, True)], 0.5) == 1.0

    def test_none_when_class_absent(self):
        assert metrics.recall_at([(0.9, False)], 0.5) is None
        assert metrics.false_positive_rate([(0.9, True)], 0.5) is None


class TestEqualErrorRate:
    def test_perfect_detector_has_zero_eer(self):
        pairs = [(0.0, False), (0.1, False), (0.9, True), (1.0, True)]
        assert metrics.equal_error_rate(pairs) == pytest.approx(0.0, abs=1e-9)

    def test_random_detector_has_high_eer(self):
        pairs = [(0.5, True), (0.5, False)]
        assert metrics.equal_error_rate(pairs) >= 0.5

    def test_none_when_single_class(self):
        assert metrics.equal_error_rate([(0.5, True)]) is None


class TestPerGroupRecall:
    def _rows(self):
        return [
            {"label": "fake", "generator": "sora", "verdict": False},
            {"label": "fake", "generator": "sora", "verdict": False},
            {"label": "fake", "generator": "sd15", "verdict": True},
            {"label": "fake", "generator": "sd15", "verdict": True},
            {"label": "real", "generator": "coco", "verdict": False},
        ]

    def test_sorted_worst_first(self):
        result = metrics.per_group_recall(self._rows(), "generator")
        assert result[0][0] == "sora"
        assert result[0][1] == 0.0
        assert result[-1][0] == "sd15"
        assert result[-1][1] == 1.0

    def test_ignores_real_samples(self):
        """Recall is defined on positives; real media must not appear."""
        groups = {g for g, _, _ in metrics.per_group_recall(self._rows(), "generator")}
        assert "coco" not in groups

    def test_min_count_filters_small_groups(self):
        assert metrics.per_group_recall(self._rows(), "generator", min_count=3) == []
