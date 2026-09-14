"""Comprehensive metric computations for deepfake detection evaluation.

All functions accept plain Python lists of labels (0/1) and probabilities
(float 0-1) -- no framework dependencies beyond numpy and sklearn.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_eer(labels: list, probs: list) -> tuple:
    """Compute Equal Error Rate and the threshold where FPR == FNR.

    Args:
        labels: Ground truth (0=real, 1=fake).
        probs: Predicted fake probabilities.

    Returns:
        Tuple of (eer, threshold).
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    # Find the point where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, thresh


def compute_fpr_at_fnr(labels: list, probs: list, target_fnr: float = 0.01) -> float:
    """Compute FPR at a fixed FNR (e.g., FPR@1%FNR).

    Args:
        labels: Ground truth.
        probs: Predicted probabilities.
        target_fnr: Target false negative rate.

    Returns:
        FPR at the given FNR level.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    # Find the FPR where FNR is closest to target
    idx = np.nanargmin(np.abs(fnr - target_fnr))
    return float(fpr[idx])


def compute_fnr_at_fpr(labels: list, probs: list, target_fpr: float = 0.01) -> float:
    """Compute FNR at a fixed FPR (e.g., FNR@1%FPR).

    Args:
        labels: Ground truth.
        probs: Predicted probabilities.
        target_fpr: Target false positive rate.

    Returns:
        FNR at the given FPR level.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    # Among points at or below the target FPR, pick the one with lowest FNR.
    # Falls back to closest FPR if none are at or below the target.
    feasible = np.where(fpr <= target_fpr)[0]
    if len(feasible) > 0:
        idx = feasible[np.argmin(fnr[feasible])]
    else:
        idx = np.nanargmin(np.abs(fpr - target_fpr))
    return float(fnr[idx])


def compute_ece(labels: list, probs: list, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error.

    Args:
        labels: Ground truth.
        probs: Predicted probabilities.
        n_bins: Number of bins for calibration.

    Returns:
        ECE value (0 = perfectly calibrated, 1 = worst).
    """
    labels_arr = np.array(labels)
    probs_arr = np.array(probs)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs_arr >= bin_boundaries[i]) & (probs_arr < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels_arr[mask].mean()
        bin_conf = probs_arr[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(labels)) if len(labels) > 0 else 0.0


def bootstrap_auc_ci(
    labels: list,
    probs: list,
    n_iter: int = 1000,
    ci: float = 0.95,
) -> tuple:
    """Compute AUC with bootstrap confidence interval.

    Args:
        labels: Ground truth.
        probs: Predicted probabilities.
        n_iter: Number of bootstrap iterations.
        ci: Confidence level (default 95%).

    Returns:
        Tuple of (point_auc, (lower_bound, upper_bound)).
    """
    labels_arr = np.array(labels)
    probs_arr = np.array(probs)
    n = len(labels_arr)

    point_auc = float(roc_auc_score(labels_arr, probs_arr))

    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        boot_labels = labels_arr[idx]
        boot_probs = probs_arr[idx]
        if len(set(boot_labels)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(boot_labels, boot_probs))
        except ValueError:
            continue

    if not aucs:
        return point_auc, (point_auc, point_auc)

    alpha = (1 - ci) / 2
    lo = float(np.percentile(aucs, 100 * alpha))
    hi = float(np.percentile(aucs, 100 * (1 - alpha)))
    return point_auc, (lo, hi)


def compute_per_generator_auc(predictions: list) -> dict:
    """Compute AUC per fake generator using global real pool.

    The 0.0 AUC bug happens when computing AUC per-generator without
    enough real samples in that generator bucket. Fix: pool ALL real
    samples as the negative class for every generator.

    Args:
        predictions: List of dicts with 'label', 'generator',
            'ensemble_prob'.

    Returns:
        Dict of generator_name -> AUC.
    """
    # Collect all real samples as global negative pool
    real_probs = [
        p["ensemble_prob"]
        for p in predictions
        if p["label"] == 0 and p["ensemble_prob"] is not None
    ]
    real_labels = [0] * len(real_probs)

    # Per-generator: combine that generator's fakes with all reals
    gen_fakes = defaultdict(list)
    for p in predictions:
        if p["label"] == 1 and p["ensemble_prob"] is not None:
            gen_fakes[p["generator"]].append(p["ensemble_prob"])

    result = {}
    for gen, fake_probs in gen_fakes.items():
        if len(fake_probs) < 1:
            continue
        combined_labels = real_labels + [1] * len(fake_probs)
        combined_probs = real_probs + fake_probs
        if len(set(combined_labels)) < 2:
            continue
        try:
            auc = float(roc_auc_score(combined_labels, combined_probs))
            result[gen] = round(auc, 4)
        except ValueError:
            result[gen] = None

    return result


def compute_all_metrics(labels: list, probs: list, threshold: float = 0.5) -> dict:
    """Compute all metrics from labels and probabilities.

    Args:
        labels: Ground truth (0=real, 1=fake).
        probs: Predicted fake probabilities.
        threshold: Classification threshold.

    Returns:
        Dict with all metrics.
    """
    labels_arr = np.array(labels)
    probs_arr = np.array(probs)
    preds = (probs_arr >= threshold).astype(int)

    result = {}

    # Standard metrics
    result["auc_roc"] = round(float(roc_auc_score(labels_arr, probs_arr)), 4)
    result["auc_pr"] = round(float(average_precision_score(labels_arr, probs_arr)), 4)
    result["accuracy"] = round(float(accuracy_score(labels_arr, preds)), 4)
    result["precision"] = round(
        float(precision_score(labels_arr, preds, zero_division=0)), 4
    )
    result["recall"] = round(float(recall_score(labels_arr, preds, zero_division=0)), 4)
    result["f1"] = round(float(f1_score(labels_arr, preds, zero_division=0)), 4)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels_arr, preds, labels=[0, 1]).ravel()
    result["tp"] = int(tp)
    result["tn"] = int(tn)
    result["fp"] = int(fp)
    result["fn"] = int(fn)

    # Optimal threshold (Youden's J)
    fpr_arr, tpr_arr, thresholds = roc_curve(labels_arr, probs_arr)
    j_scores = tpr_arr - fpr_arr
    best_idx = j_scores.argmax()
    result["optimal_threshold"] = round(
        float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5,
        4,
    )

    # EER
    eer, eer_thresh = compute_eer(labels, probs)
    result["eer"] = round(eer, 4)
    result["eer_threshold"] = round(eer_thresh, 4)

    # FPR@FNR
    result["fpr_at_1pct_fnr"] = round(compute_fpr_at_fnr(labels, probs, 0.01), 4)
    result["fpr_at_5pct_fnr"] = round(compute_fpr_at_fnr(labels, probs, 0.05), 4)
    result["fnr_at_1pct_fpr"] = round(compute_fnr_at_fpr(labels, probs, 0.01), 4)
    result["fnr_at_5pct_fpr"] = round(compute_fnr_at_fpr(labels, probs, 0.05), 4)

    # Calibration
    result["ece"] = round(compute_ece(labels, probs), 4)

    # Bootstrap CI
    point_auc, (ci_lo, ci_hi) = bootstrap_auc_ci(labels, probs, n_iter=1000)
    result["bootstrap_ci"] = {
        "auc": round(point_auc, 4),
        "ci_95_lower": round(ci_lo, 4),
        "ci_95_upper": round(ci_hi, 4),
    }

    result["n_total"] = len(labels)
    result["n_real"] = int(sum(1 for l in labels if l == 0))
    result["n_fake"] = int(sum(1 for l in labels if l == 1))

    return result
