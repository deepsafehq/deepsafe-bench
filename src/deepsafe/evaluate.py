"""The ``deepsafe eval`` verb: score a detector against the benchmark."""

from __future__ import annotations

import pathlib
from typing import Callable, Optional

from deepsafe import data, metrics
from deepsafe.report import ReportCard

DEFAULT_THRESHOLD = 0.5
WORST_N = 10
BEST_N = 5


def evaluate_baseline(
    model: str,
    *,
    modality: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    predictions: Optional[pathlib.Path] = None,
) -> ReportCard:
    """Score a model already present in the stored prediction matrix.

    This reproduces published numbers without downloading any media, which
    makes the headline results checkable by anyone in seconds.

    Args:
        model: Model name, e.g. ``"npr"``, or ``"ensemble"``.
        modality: Restrict to ``images``, ``audio``, or ``video``.
        threshold: Decision threshold for recall and FPR.
        predictions: Override the prediction matrix path.

    Returns:
        A populated ReportCard.

    Raises:
        KeyError: If the model has no column in the matrix.
    """
    rows = data.load_predictions(predictions)
    column = "ensemble_score" if model == "ensemble" else f"prob_{model}"
    if column not in rows[0]:
        available = sorted(
            c[len("prob_"):] for c in rows[0] if c.startswith("prob_")
        )
        raise KeyError(
            f"no scores for {model!r} in the prediction matrix.\n"
            f"Available: {', '.join(available)}, ensemble"
        )

    scored = [r for r in rows if data.as_float(r, column) is not None]
    if modality:
        scored = [r for r in scored if r.get("modality") == modality]
    if not scored:
        raise KeyError(f"no rows for {model!r} with modality={modality!r}")

    pairs = [(data.as_float(r, column), r["label"] == "fake") for r in scored]
    graded = [
        {
            "generator": r.get("generator", "unknown"),
            "label": r.get("label"),
            "verdict": (data.as_float(r, column) or 0.0) >= threshold,
        }
        for r in scored
    ]
    ranked = metrics.per_group_recall(graded, "generator", min_count=30)

    return ReportCard(
        detector=model,
        tier=f"stored predictions{f' / {modality}' if modality else ''}",
        n_samples=len(scored),
        auc=metrics.roc_auc(pairs),
        recall=metrics.recall_at(pairs, threshold),
        fpr=metrics.false_positive_rate(pairs, threshold),
        eer=metrics.equal_error_rate(pairs),
        threshold=threshold,
        worst_generators=ranked[:WORST_N],
        best_generators=list(reversed(ranked[-BEST_N:])),
        models_used=[model] if model != "ensemble" else [],
    )


def evaluate_detector(
    predict: Callable[[pathlib.Path], float],
    *,
    name: str,
    tier: str,
    root: pathlib.Path,
    threshold: float = DEFAULT_THRESHOLD,
    limit: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> ReportCard:
    """Score a user-supplied detector against a dataset tier.

    A detector that raises on a sample is recorded as a failure on that sample
    rather than aborting the run; a model that crashes on 30% of inputs should
    show up in its score, not as a stack trace.

    Args:
        predict: Callable returning P(synthetic).
        name: Display name for the report.
        tier: ``small``, ``medium``, or ``full``.
        root: Directory containing the tier.
        threshold: Decision threshold.
        limit: Stop after this many samples.
        on_progress: Called with ``(done, total)``.

    Returns:
        A populated ReportCard.
    """
    samples = list(data.iter_samples(tier, root))
    if limit:
        samples = samples[:limit]

    pairs: list[tuple[Optional[float], bool]] = []
    graded: list[dict] = []
    errors = 0

    for index, (media, record) in enumerate(samples, start=1):
        try:
            score = float(predict(media))
        except Exception:
            score = 0.0  # a crash is a miss, not an excuse
            errors += 1
        is_fake = record.get("label") == "fake"
        pairs.append((score, is_fake))
        graded.append(
            {
                "generator": record.get("generator", "unknown"),
                "label": record.get("label"),
                "verdict": score >= threshold,
            }
        )
        if on_progress:
            on_progress(index, len(samples))

    ranked = metrics.per_group_recall(graded, "generator", min_count=5)
    card = ReportCard(
        detector=name,
        tier=tier,
        n_samples=len(samples),
        auc=metrics.roc_auc(pairs),
        recall=metrics.recall_at(pairs, threshold),
        fpr=metrics.false_positive_rate(pairs, threshold),
        eer=metrics.equal_error_rate(pairs),
        threshold=threshold,
        worst_generators=ranked[:WORST_N],
        best_generators=list(reversed(ranked[-BEST_N:])),
        models_used=[],
    )
    if errors:
        card.detector = f"{name} ({errors} sample(s) raised, counted as misses)"
    return card
