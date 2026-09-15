"""Scoring primitives for the DeepSafe benchmark.

Implemented without numpy, pandas, or scikit-learn on purpose: the published
benchmark numbers must be reproducible on a bare Python install, and a reader
checking our arithmetic should not have to trust a dependency to do it.
"""

from __future__ import annotations

import collections
from typing import Iterable, Optional, Sequence


def roc_auc(pairs: Iterable[tuple[float, bool]]) -> Optional[float]:
    """Compute ROC-AUC via the Mann-Whitney U statistic.

    Ties receive averaged ranks, which matters because several detectors
    saturate at 0.0 or 1.0 on large slices of the evaluation set.

    Args:
        pairs: ``(score, is_positive)`` tuples. ``None`` scores are dropped.

    Returns:
        The AUC in [0, 1], or None when either class is empty.
    """
    scored = [(s, y) for s, y in pairs if s is not None]
    positives = sum(1 for _, y in scored if y)
    negatives = len(scored) - positives
    if not positives or not negatives:
        return None

    order = sorted(scored, key=lambda t: t[0])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1

    rank_sum = sum(ranks[k] for k, (_, y) in enumerate(order) if y)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def recall_at(pairs: Iterable[tuple[float, bool]], threshold: float) -> Optional[float]:
    """Fraction of positives scored at or above ``threshold``."""
    positives = [s for s, y in pairs if y and s is not None]
    if not positives:
        return None
    return sum(1 for s in positives if s >= threshold) / len(positives)


def false_positive_rate(
    pairs: Iterable[tuple[float, bool]], threshold: float
) -> Optional[float]:
    """Fraction of negatives wrongly scored at or above ``threshold``."""
    negatives = [s for s, y in pairs if not y and s is not None]
    if not negatives:
        return None
    return sum(1 for s in negatives if s >= threshold) / len(negatives)


def equal_error_rate(pairs: Iterable[tuple[float, bool]]) -> Optional[float]:
    """Return the equal error rate, where FPR and FNR cross.

    Scans every score as a candidate threshold and returns the smallest
    ``max(fpr, fnr)``, which coincides with the EER at the crossing point.
    """
    scored = [(s, y) for s, y in pairs if s is not None]
    if not scored or all(y for _, y in scored) or not any(y for _, y in scored):
        return None

    best = 1.0
    for threshold in sorted({s for s, _ in scored}):
        fpr = false_positive_rate(scored, threshold) or 0.0
        fnr = 1.0 - (recall_at(scored, threshold) or 0.0)
        best = min(best, max(fpr, fnr))
    return best


def per_group_recall(
    rows: Sequence[dict],
    group_key: str,
    *,
    min_count: int = 1,
) -> list[tuple[str, float, int]]:
    """Detection rate on positives, grouped by a metadata field.

    This is the view that exposes generalization failure: an aggregate number
    hides that a detector is perfect on one generator and blind on another.

    Args:
        rows: Records with ``label``, ``verdict``, and ``group_key`` fields.
        group_key: Field to group by, typically ``"generator"``.
        min_count: Drop groups with fewer than this many positives.

    Returns:
        ``(group, recall, n)`` tuples sorted worst-first.
    """
    buckets: dict[str, list[bool]] = collections.defaultdict(list)
    for row in rows:
        if row.get("label") != "fake" or row.get("verdict") is None:
            continue
        buckets[str(row.get(group_key, "unknown"))].append(bool(row["verdict"]))

    out = [
        (group, sum(hits) / len(hits), len(hits))
        for group, hits in buckets.items()
        if len(hits) >= min_count
    ]
    return sorted(out, key=lambda t: t[1])
