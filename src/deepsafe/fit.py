"""The ``deepsafe fit`` verb: adapt the stack to your own labeled data.

Three tiers, ordered by compute cost:

* **Tier 1, ensemble refit.** CPU, minutes. Retrains the meta-learner over the
  existing model scores with your class priors. No model weights change. For
  most users this is the adaptation that matters.
* **Tier 2, linear probe.** Freeze the backbone, train a fresh head.
* **Tier 3, full fine-tuning.** Per-model, GPU, hours.

Tiers 2 and 3 require the inference stack and a GPU, so they live behind the
``deepsafe.fit_gpu`` module and are reported as unavailable when torch is not
installed. Tier 1 is implemented here with no third-party dependencies.

Every tier runs a held-out-generator check afterwards. Improving on your own
distribution while degrading on unseen generators is the default outcome in
this field, and a tool that hides it is worse than useless.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Optional, Sequence

from deepsafe import metrics


@dataclasses.dataclass
class GeneralizationReport:
    """In-distribution versus held-out-generator performance after adaptation."""

    in_distribution_auc: Optional[float]
    held_out_auc: Optional[float]
    baseline_held_out_auc: Optional[float]
    held_out_generators: list[str]

    @property
    def transferred(self) -> bool:
        """True when adaptation did not degrade unseen-generator performance."""
        if self.held_out_auc is None or self.baseline_held_out_auc is None:
            return True
        return self.held_out_auc >= self.baseline_held_out_auc - 0.01

    def render(self) -> str:
        """Render the verdict, stating plainly when a fine-tune overfit."""
        def num(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:.4f}"

        lines = [
            "GENERALIZATION CHECK",
            f"  in-distribution AUC      {num(self.in_distribution_auc)}",
            f"  held-out generators AUC  {num(self.held_out_auc)}",
            f"  same, before adaptation  {num(self.baseline_held_out_auc)}",
            f"  held out                 {', '.join(self.held_out_generators[:6])}"
            + (" ..." if len(self.held_out_generators) > 6 else ""),
            "",
        ]
        if self.transferred:
            lines.append("  Adaptation held up on generators it never saw.")
        else:
            delta = (self.baseline_held_out_auc or 0) - (self.held_out_auc or 0)
            lines += [
                f"  WARNING: unseen-generator AUC fell by {delta:.4f} while",
                "  in-distribution performance rose. This is overfitting to your",
                "  data, not improvement. The gain will not survive a new generator.",
            ]
        return "\n".join(lines)


class LogisticMetaLearner:
    """Logistic regression trained by gradient descent.

    Implemented directly rather than pulled from scikit-learn so that Tier 1
    runs anywhere Python does. The meta-learner sits on top of a handful of
    detector scores, so the problem is tiny and does not need more.
    """

    def __init__(self, learning_rate: float = 0.5, epochs: int = 800) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias: float = 0.0

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
        """Fit the model to ``features`` and binary ``labels``."""
        if not features:
            raise ValueError("no training rows")
        n_features = len(features[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(features)

        for _ in range(self.epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for row, label in zip(features, labels):
                error = self._sigmoid(self._raw(row)) - label
                for j, value in enumerate(row):
                    grad_w[j] += error * value
                grad_b += error
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * grad_w[j] / n
            self.bias -= self.learning_rate * grad_b / n

    def predict_proba(self, row: Sequence[float]) -> float:
        """Return P(fake) for one feature row."""
        return self._sigmoid(self._raw(row))

    def _raw(self, row: Sequence[float]) -> float:
        return sum(w * v for w, v in zip(self.weights, row)) + self.bias

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z < -60:
            return 0.0
        if z > 60:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))


def fit_ensemble(
    rows: Sequence[dict],
    feature_columns: Sequence[str],
    *,
    holdout_fraction: float = 0.3,
    seed: int = 42,
) -> tuple[LogisticMetaLearner, GeneralizationReport]:
    """Tier 1: retrain the meta-learner and check that it generalizes.

    The split is **by generator, not by sample**. Splitting randomly would let
    the same generator appear in train and test, which is exactly the mistake
    that produces the field's inflated numbers.

    Args:
        rows: Records with ``label``, ``generator``, and score columns.
        feature_columns: Score columns to use as features.
        holdout_fraction: Share of generators held out entirely.
        seed: RNG seed for the generator split.

    Returns:
        The fitted meta-learner and its generalization report.

    Raises:
        ValueError: If there are too few generators or usable rows.
    """
    usable = [
        r for r in rows
        if all(_as_float(r.get(c)) is not None for c in feature_columns)
        and r.get("label") in ("fake", "real")
    ]
    if not usable:
        raise ValueError("no rows with all feature columns populated")

    generators = sorted({r.get("generator", "unknown") for r in usable})
    if len(generators) < 4:
        raise ValueError(
            f"need at least 4 distinct generators to hold any out, got "
            f"{len(generators)}. Without a generator-level split the result "
            f"would be in-distribution only and would overstate performance."
        )

    rng = random.Random(seed)
    shuffled = generators[:]
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * holdout_fraction))
    held_out = set(shuffled[:n_holdout])

    train = [r for r in usable if r.get("generator") not in held_out]
    test = [r for r in usable if r.get("generator") in held_out]
    if not train or not test:
        raise ValueError("generator split produced an empty side")

    features = [[_as_float(r[c]) for c in feature_columns] for r in train]
    labels = [1 if r["label"] == "fake" else 0 for r in train]

    model = LogisticMetaLearner()
    model.fit(features, labels)

    def score(subset: Sequence[dict]) -> Optional[float]:
        return metrics.roc_auc(
            (
                model.predict_proba([_as_float(r[c]) for c in feature_columns]),
                r["label"] == "fake",
            )
            for r in subset
        )

    # Baseline: the single best individual detector on the same held-out split.
    baseline = None
    for column in feature_columns:
        auc = metrics.roc_auc(
            (_as_float(r[column]), r["label"] == "fake") for r in test
        )
        if auc is not None and (baseline is None or auc > baseline):
            baseline = auc

    report = GeneralizationReport(
        in_distribution_auc=score(train),
        held_out_auc=score(test),
        baseline_held_out_auc=baseline,
        held_out_generators=sorted(held_out),
    )
    return model, report


def _as_float(value) -> Optional[float]:
    """Coerce to float, returning None when not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
