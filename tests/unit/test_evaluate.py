"""Tests for the eval verb against the bundled prediction matrix."""

import pytest

from deepsafe.evaluate import evaluate_baseline, evaluate_detector
from deepsafe.report import ReportCard


class TestEvaluateBaseline:
    def test_reproduces_published_video_ensemble_auc(self):
        """The headline number in BENCHMARK.md must stay reproducible."""
        card = evaluate_baseline("ensemble", modality="video")
        assert card.auc == pytest.approx(0.6694, abs=5e-4)

    def test_reproduces_published_image_ensemble_auc(self):
        card = evaluate_baseline("ensemble", modality="images")
        assert card.auc == pytest.approx(0.9466, abs=5e-4)

    def test_sora_recall_stays_catastrophic(self):
        """If this ever passes silently, the benchmark has been corrupted."""
        card = evaluate_baseline("ensemble", modality="video")
        worst = dict((g, r) for g, r, _ in card.worst_generators)
        assert worst["sora"] < 0.10

    def test_unknown_model_lists_alternatives(self):
        with pytest.raises(KeyError, match="Available"):
            evaluate_baseline("not_a_model")

    def test_report_separates_best_and_worst(self):
        card = evaluate_baseline("ensemble", modality="video")
        assert card.worst_generators[0][1] <= card.best_generators[0][1]


class TestEvaluateDetector:
    def test_a_crashing_detector_scores_rather_than_aborts(self, tmp_path, monkeypatch):
        """A model that raises on 100% of inputs must score, not explode."""
        import deepsafe.data as data

        samples = [
            (tmp_path / "a.jpg", {"label": "fake", "generator": "g"}),
            (tmp_path / "b.jpg", {"label": "real", "generator": "coco"}),
        ]
        monkeypatch.setattr(data, "iter_samples", lambda *a, **k: iter(samples))
        monkeypatch.setattr("deepsafe.evaluate.data.iter_samples",
                            lambda *a, **k: iter(samples))

        def exploding(path):
            raise RuntimeError("boom")

        card = evaluate_detector(exploding, name="bad", tier="small", root=tmp_path)
        assert isinstance(card, ReportCard)
        assert "raised" in card.detector
        assert card.n_samples == 2


class TestReportCard:
    def test_citations_include_models_used(self):
        card = evaluate_baseline("npr", modality="images")
        bib = card.citations()
        assert "deepsafe2026" in bib
        assert "npr2024" in bib

    def test_render_states_the_spread(self):
        card = evaluate_baseline("ensemble", modality="video")
        assert "Spread between best and worst generator" in card.render()

    def test_write_emits_both_files(self, tmp_path):
        card = evaluate_baseline("npr", modality="images")
        report, bib = card.write(tmp_path)
        assert report.exists() and bib.exists()
        assert "REPORT CARD" in report.read_text()
