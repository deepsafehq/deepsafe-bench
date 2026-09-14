#!/usr/bin/env python3
"""Ensemble R&D lab: compare meta-learner architectures, run ablation
studies, and optimize decision thresholds.

Consumes ``eval/results/{modality}_predictions.json`` and writes
``eval/results/{modality}_ensemble_rd.json`` with structured experiment
results for RF, XGBoost, LightGBM, logistic regression, weighted voting,
single/multi-model ablation, threshold optimization, and prior-adjusted
precision (Bayes' theorem).

All ensemble experiments use StratifiedKFold 5-fold CV with seed 42 and
out-of-fold ``cross_val_predict`` probabilities.

Usage:
    python ensemble_lab.py --modality image
    python ensemble_lab.py --modality audio
    python ensemble_lab.py --modality video
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                 n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        ECE value in [0, 1].
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(y_true)) if len(y_true) > 0 else 0.0


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC-ROC with graceful fallback for degenerate cases.

    Args:
        y: Binary ground truth.
        p: Predicted probabilities.

    Returns:
        AUC-ROC value, or 0.5 if computation fails.
    """
    if len(set(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(modality: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load predictions JSON and build feature matrix.

    Drops rows where any model has a null probability.

    Args:
        modality: One of ``image``, ``audio``, ``video``.

    Returns:
        Tuple of (feature DataFrame, label array, model name list).

    Raises:
        SystemExit: If the file is missing or yields no usable rows.
    """
    path = RESULTS_DIR / f"{modality}_predictions.json"
    if not path.exists():
        print(f"ERROR: Predictions file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "predictions" in raw:
        preds = raw["predictions"]
    elif isinstance(raw, list):
        preds = raw
    else:
        print("ERROR: Unrecognised predictions format.", file=sys.stderr)
        sys.exit(1)

    # Discover model names and filter out those with zero valid predictions.
    model_valid_count: dict[str, int] = {}
    for p in preds:
        mp = p.get("model_results") or {}
        for m, mr in mp.items():
            prob = mr.get("probability") if isinstance(mr, dict) else mr
            if prob is not None:
                model_valid_count[m] = model_valid_count.get(m, 0) + 1
    models = sorted(m for m, count in model_valid_count.items() if count > 0)
    if not models:
        print("ERROR: No models with valid predictions.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(models)} models with data: {models}")

    # Build matrix -- keep only rows where ALL models are present.
    rows = []
    labels = []
    for p in preds:
        mp = p.get("model_results") or {}
        # Extract probability from each model's result dict
        probs = {}
        all_valid = True
        for m in models:
            mr = mp.get(m)
            if mr is None:
                all_valid = False
                break
            prob = mr.get("probability") if isinstance(mr, dict) else mr
            if prob is None:
                all_valid = False
                break
            probs[m] = prob
        if all_valid:
            rows.append(probs)
            labels.append(p.get("label_int", p.get("label")))

    if not rows:
        print("ERROR: No files with all models valid.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    y = np.array(labels, dtype=int)

    print(f"  Loaded {len(df)} samples x {len(models)} models")
    print(f"  Labels: real={int((y == 0).sum())}, fake={int((y == 1).sum())}")
    return df, y, models


# ---------------------------------------------------------------------------
# Experiment: RF variants
# ---------------------------------------------------------------------------

def _run_rf_variants(
    x: np.ndarray,
    y: np.ndarray,
    skf: StratifiedKFold,
) -> list[dict]:
    """Train RF classifiers with different hyperparameter configs.

    Configs: (n_estimators, max_depth) =
    (300, 8), (500, 10), (500, 12), (700, 14).

    Args:
        x: Scaled feature matrix.
        y: Binary labels.
        skf: Pre-configured StratifiedKFold splitter.

    Returns:
        List of result dicts with ``config``, ``cv_auc``, ``cv_ece``.
    """
    configs = [
        (300, 8),
        (500, 10),
        (500, 12),
        (700, 14),
    ]
    results = []
    for n_est, depth in configs:
        tag = f"rf_{n_est}_d{depth}"
        print(f"  {tag}...")
        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            random_state=42,
            n_jobs=-1,
        )
        proba = cross_val_predict(
            clf, x, y, cv=skf, method="predict_proba"
        )[:, 1]
        auc = round(_safe_auc(y, proba), 4)
        ece = round(_compute_ece(y, proba), 4)
        results.append({
            "name": tag,
            "config": {"n_estimators": n_est, "max_depth": depth},
            "cv_auc": auc,
            "cv_ece": ece,
        })
        print(f"    AUC={auc}  ECE={ece}")
    return results


# ---------------------------------------------------------------------------
# Experiment: XGBoost
# ---------------------------------------------------------------------------

def _run_xgboost(
    x: np.ndarray,
    y: np.ndarray,
    skf: StratifiedKFold,
) -> list[dict]:
    """Train XGBoost classifiers if available.

    Configs: (n_estimators, max_depth, learning_rate) =
    (200, 4, 0.1), (300, 6, 0.05), (500, 4, 0.05).

    Args:
        x: Scaled feature matrix.
        y: Binary labels.
        skf: Pre-configured StratifiedKFold splitter.

    Returns:
        List of result dicts, or a single dict with ``skipped: True``.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  XGBoost not installed -- skipping.")
        return [{"name": "xgboost", "skipped": True,
                 "reason": "xgboost not installed"}]

    configs = [
        (200, 4, 0.1),
        (300, 6, 0.05),
        (500, 4, 0.05),
    ]
    results = []
    for n_est, depth, lr in configs:
        tag = f"xgb_{n_est}_d{depth}_lr{lr}"
        print(f"  {tag}...")
        clf = XGBClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        proba = cross_val_predict(
            clf, x, y, cv=skf, method="predict_proba"
        )[:, 1]
        auc = round(_safe_auc(y, proba), 4)
        ece = round(_compute_ece(y, proba), 4)
        results.append({
            "name": tag,
            "config": {"n_estimators": n_est, "max_depth": depth,
                       "learning_rate": lr},
            "cv_auc": auc,
            "cv_ece": ece,
        })
        print(f"    AUC={auc}  ECE={ece}")
    return results


# ---------------------------------------------------------------------------
# Experiment: LightGBM
# ---------------------------------------------------------------------------

def _run_lightgbm(
    x: np.ndarray,
    y: np.ndarray,
    skf: StratifiedKFold,
) -> list[dict]:
    """Train LightGBM classifiers if available.

    Configs: (n_estimators, max_depth, learning_rate) =
    (200, 4, 0.1), (300, 6, 0.05).

    Args:
        x: Scaled feature matrix.
        y: Binary labels.
        skf: Pre-configured StratifiedKFold splitter.

    Returns:
        List of result dicts, or a single dict with ``skipped: True``.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("  LightGBM not installed -- skipping.")
        return [{"name": "lightgbm", "skipped": True,
                 "reason": "lightgbm not installed"}]

    configs = [
        (200, 4, 0.1),
        (300, 6, 0.05),
    ]
    results = []
    for n_est, depth, lr in configs:
        tag = f"lgbm_{n_est}_d{depth}_lr{lr}"
        print(f"  {tag}...")
        clf = LGBMClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        proba = cross_val_predict(
            clf, x, y, cv=skf, method="predict_proba"
        )[:, 1]
        auc = round(_safe_auc(y, proba), 4)
        ece = round(_compute_ece(y, proba), 4)
        results.append({
            "name": tag,
            "config": {"n_estimators": n_est, "max_depth": depth,
                       "learning_rate": lr},
            "cv_auc": auc,
            "cv_ece": ece,
        })
        print(f"    AUC={auc}  ECE={ece}")
    return results


# ---------------------------------------------------------------------------
# Experiment: Logistic Regression
# ---------------------------------------------------------------------------

def _run_logistic(
    x: np.ndarray,
    y: np.ndarray,
    skf: StratifiedKFold,
) -> dict:
    """StandardScaler + LogisticRegression baseline.

    Args:
        x: Raw (unscaled) feature matrix.
        y: Binary labels.
        skf: Pre-configured StratifiedKFold splitter.

    Returns:
        Result dict with ``cv_auc`` and ``cv_ece``.
    """
    print("  logistic_regression...")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    proba = cross_val_predict(
        clf, x_scaled, y, cv=skf, method="predict_proba"
    )[:, 1]
    auc = round(_safe_auc(y, proba), 4)
    ece = round(_compute_ece(y, proba), 4)
    print(f"    AUC={auc}  ECE={ece}")
    return {"name": "logistic_regression", "cv_auc": auc, "cv_ece": ece}


# ---------------------------------------------------------------------------
# Experiment: Weighted voting
# ---------------------------------------------------------------------------

def _run_weighted_voting(
    df: pd.DataFrame,
    y: np.ndarray,
    models: list[str],
) -> dict:
    """AUC-weighted average of raw model probabilities.

    Weight for each model = max(individual_AUC - 0.5, 0).

    Args:
        df: Feature DataFrame (raw probabilities).
        y: Binary labels.
        models: Model name list.

    Returns:
        Result dict with per-model weights, ``auc``, and ``ece``.
    """
    print("  weighted_voting...")
    per_model_auc = {}
    for m in models:
        per_model_auc[m] = _safe_auc(y, df[m].values)

    weights = {m: max(auc - 0.5, 0.0) for m, auc in per_model_auc.items()}
    total_w = sum(weights.values())
    if total_w == 0:
        total_w = 1.0  # Fallback to uniform.

    proba = np.zeros(len(df))
    for m in models:
        proba += (weights[m] / total_w) * df[m].values

    auc = round(_safe_auc(y, proba), 4)
    ece = round(_compute_ece(y, proba), 4)
    print(f"    AUC={auc}  ECE={ece}")
    return {
        "name": "weighted_voting",
        "per_model_auc": {m: round(v, 4) for m, v in per_model_auc.items()},
        "weights": {m: round(v, 4) for m, v in weights.items()},
        "auc": auc,
        "ece": ece,
    }


# ---------------------------------------------------------------------------
# Experiment: Single-model ablation
# ---------------------------------------------------------------------------

def _run_single_ablation(
    x: np.ndarray,
    y: np.ndarray,
    models: list[str],
    best_rf_config: tuple[int, int],
    skf: StratifiedKFold,
) -> dict:
    """Drop each model one at a time, retrain best RF, measure AUC delta.

    Verdict thresholds:
    - KEEP: delta < -0.005 (dropping hurts significantly)
    - MARGINAL: -0.005 <= delta < -0.001
    - SAFE_TO_DROP: delta >= -0.001

    Args:
        x: Scaled feature matrix (all models).
        y: Binary labels.
        models: Model name list.
        best_rf_config: (n_estimators, max_depth) of the best RF.
        skf: Pre-configured StratifiedKFold splitter.

    Returns:
        Dict with ``baseline_auc``, per-model ``delta`` and
        ``verdict``, and ``safe_to_drop`` list.
    """
    print("  single-model ablation...")
    n_est, depth = best_rf_config

    # Baseline: all models.
    clf_full = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth,
        random_state=42, n_jobs=-1,
    )
    proba_full = cross_val_predict(
        clf_full, x, y, cv=skf, method="predict_proba"
    )[:, 1]
    baseline_auc = _safe_auc(y, proba_full)
    print(f"    baseline AUC={baseline_auc:.4f}")

    ablations = {}
    safe_to_drop = []
    for i, model in enumerate(models):
        # Drop column i.
        x_reduced = np.delete(x, i, axis=1)
        clf_red = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth,
            random_state=42, n_jobs=-1,
        )
        proba_red = cross_val_predict(
            clf_red, x_reduced, y, cv=skf, method="predict_proba"
        )[:, 1]
        reduced_auc = _safe_auc(y, proba_red)
        delta = reduced_auc - baseline_auc

        if delta < -0.005:
            verdict = "KEEP"
        elif delta < -0.001:
            verdict = "MARGINAL"
        else:
            verdict = "SAFE_TO_DROP"
            safe_to_drop.append(model)

        ablations[model] = {
            "reduced_auc": round(reduced_auc, 4),
            "delta": round(delta, 4),
            "verdict": verdict,
        }
        print(f"    drop {model}: AUC={reduced_auc:.4f}  "
              f"delta={delta:+.4f}  -> {verdict}")

    return {
        "baseline_auc": round(baseline_auc, 4),
        "best_rf_config": {"n_estimators": n_est, "max_depth": depth},
        "per_model": ablations,
        "safe_to_drop": safe_to_drop,
    }


# ---------------------------------------------------------------------------
# Experiment: Multi-model ablation
# ---------------------------------------------------------------------------

def _run_multi_ablation(
    x: np.ndarray,
    y: np.ndarray,
    models: list[str],
    safe_to_drop: list[str],
    best_rf_config: tuple[int, int],
    baseline_auc: float,
    skf: StratifiedKFold,
) -> dict:
    """Drop all SAFE_TO_DROP models together, measure combined impact.

    Args:
        x: Scaled feature matrix.
        y: Binary labels.
        models: Full model name list.
        safe_to_drop: Models tagged as SAFE_TO_DROP.
        best_rf_config: (n_estimators, max_depth).
        baseline_auc: AUC with all models.
        skf: Pre-configured StratifiedKFold splitter.

    Returns:
        Result dict with ``dropped_models``, ``reduced_auc``,
        ``combined_delta``, and ``remaining_models``.
    """
    print("  multi-model ablation...")
    if not safe_to_drop:
        print("    No SAFE_TO_DROP models -- skipping.")
        return {
            "dropped_models": [],
            "remaining_models": models,
            "reduced_auc": round(baseline_auc, 4),
            "combined_delta": 0.0,
        }

    drop_indices = [i for i, m in enumerate(models) if m in safe_to_drop]
    x_reduced = np.delete(x, drop_indices, axis=1)
    remaining = [m for m in models if m not in safe_to_drop]

    n_est, depth = best_rf_config
    clf = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth,
        random_state=42, n_jobs=-1,
    )
    proba = cross_val_predict(
        clf, x_reduced, y, cv=skf, method="predict_proba"
    )[:, 1]
    reduced_auc = _safe_auc(y, proba)
    delta = reduced_auc - baseline_auc

    print(f"    Dropped {safe_to_drop}: "
          f"AUC={reduced_auc:.4f}  delta={delta:+.4f}")
    return {
        "dropped_models": safe_to_drop,
        "remaining_models": remaining,
        "reduced_auc": round(reduced_auc, 4),
        "combined_delta": round(delta, 4),
    }


# ---------------------------------------------------------------------------
# Experiment: Threshold optimization
# ---------------------------------------------------------------------------

def _run_threshold_optimization(
    x: np.ndarray,
    y: np.ndarray,
    best_rf_config: tuple[int, int],
    skf: StratifiedKFold,
) -> dict:
    """Find optimal thresholds from the best RF's CV probabilities.

    Thresholds:
    - high_precision: precision > 0.99
    - balanced: max F1
    - high_recall: recall > 0.99

    Args:
        x: Scaled feature matrix.
        y: Binary labels.
        best_rf_config: (n_estimators, max_depth).
        skf: Pre-configured StratifiedKFold splitter.

    Returns:
        Dict with ``high_precision``, ``balanced``, and
        ``high_recall`` sub-dicts.
    """
    print("  threshold optimization...")
    n_est, depth = best_rf_config
    clf = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth,
        random_state=42, n_jobs=-1,
    )
    proba = cross_val_predict(
        clf, x, y, cv=skf, method="predict_proba"
    )[:, 1]

    result = {}

    # --- High precision (precision > 0.99) via PR curve ---
    prec, rec, pr_thresholds = precision_recall_curve(y, proba)
    # precision_recall_curve returns arrays where len(prec) = len(thr)+1.
    # Mask to thresholds where precision > 0.99.
    mask_hp = prec[:-1] > 0.99
    if mask_hp.any():
        # Pick the lowest threshold that achieves precision > 0.99 to
        # maximise recall.
        idx = np.where(mask_hp)[0][-1]  # Highest recall with prec > 0.99.
        result["high_precision"] = {
            "threshold": round(float(pr_thresholds[idx]), 4),
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
        }
    else:
        result["high_precision"] = {"threshold": None, "note": "not achievable"}

    # --- Balanced (max F1) via PR curve ---
    f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-10)
    best_f1_idx = int(f1_scores.argmax())
    result["balanced"] = {
        "threshold": round(float(pr_thresholds[best_f1_idx]), 4),
        "precision": round(float(prec[best_f1_idx]), 4),
        "recall": round(float(rec[best_f1_idx]), 4),
        "f1": round(float(f1_scores[best_f1_idx]), 4),
    }

    # --- High recall (recall > 0.99) via ROC curve ---
    fpr_arr, tpr_arr, roc_thresholds = roc_curve(y, proba)
    mask_hr = tpr_arr > 0.99
    if mask_hr.any():
        # Pick the highest threshold that still achieves recall > 0.99.
        idx = np.where(mask_hr)[0][0]
        result["high_recall"] = {
            "threshold": round(float(roc_thresholds[idx]), 4),
            "recall": round(float(tpr_arr[idx]), 4),
            "fpr": round(float(fpr_arr[idx]), 4),
        }
    else:
        result["high_recall"] = {"threshold": None, "note": "not achievable"}

    for key, val in result.items():
        if val.get("threshold") is not None:
            print(f"    {key}: threshold={val['threshold']}")
        else:
            print(f"    {key}: {val.get('note', 'N/A')}")
    return result


# ---------------------------------------------------------------------------
# Experiment: Prior-adjusted precision (Bayes' theorem)
# ---------------------------------------------------------------------------

def _run_prior_adjusted(
    x: np.ndarray,
    y: np.ndarray,
    best_rf_config: tuple[int, int],
    skf: StratifiedKFold,
) -> dict:
    """Compute PPV at various fake-prevalence priors using Bayes' theorem.

    PPV = (sensitivity * prevalence) /
          (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))

    Priors: 1%, 5%, 10%, 50%.

    Args:
        x: Scaled feature matrix.
        y: Binary labels.
        best_rf_config: (n_estimators, max_depth).
        skf: Pre-configured StratifiedKFold splitter.

    Returns:
        Dict mapping prevalence percentage to PPV.
    """
    print("  prior-adjusted precision...")
    n_est, depth = best_rf_config
    clf = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth,
        random_state=42, n_jobs=-1,
    )
    proba = cross_val_predict(
        clf, x, y, cv=skf, method="predict_proba"
    )[:, 1]

    # Measure sensitivity and specificity at threshold 0.5.
    preds = (proba >= 0.5).astype(int)
    tp = int(((preds == 1) & (y == 1)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    prevalences = [0.01, 0.05, 0.10, 0.50]
    result = {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
    }
    ppv_table = {}
    for prev in prevalences:
        numerator = sensitivity * prev
        denominator = numerator + (1 - specificity) * (1 - prev)
        ppv = numerator / denominator if denominator > 0 else 0.0
        key = f"{int(prev * 100)}%"
        ppv_table[key] = round(ppv, 4)
        print(f"    prevalence={key}: PPV={ppv:.4f}")

    result["ppv_by_prevalence"] = ppv_table
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse args, run all experiments, write JSON."""
    parser = argparse.ArgumentParser(
        description="Ensemble R&D lab: compare architectures and ablations."
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=["image", "audio", "video"],
        help="Modality to evaluate.",
    )
    args = parser.parse_args()
    modality: str = args.modality

    print(f"=== Ensemble Lab: {modality} ===")
    df, y, models = _load_data(modality)
    x_raw = df[models].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    output: dict = {
        "modality": modality,
        "n_samples": int(len(y)),
        "n_models": len(models),
        "models": models,
        "label_distribution": {
            "real": int((y == 0).sum()),
            "fake": int((y == 1).sum()),
        },
    }

    # 1. RF variants.
    print("\n[1/9] Random Forest variants")
    rf_results = _run_rf_variants(x_scaled, y, skf)
    output["rf_variants"] = rf_results

    # Identify best RF config for later experiments.
    best_rf = max(rf_results, key=lambda r: r["cv_auc"])
    best_rf_config = (
        best_rf["config"]["n_estimators"],
        best_rf["config"]["max_depth"],
    )
    output["best_rf"] = best_rf["name"]
    print(f"  Best RF: {best_rf['name']}  AUC={best_rf['cv_auc']}")

    # 2. XGBoost.
    print("\n[2/9] XGBoost variants")
    output["xgboost"] = _run_xgboost(x_scaled, y, skf)

    # 3. LightGBM.
    print("\n[3/9] LightGBM variants")
    output["lightgbm"] = _run_lightgbm(x_scaled, y, skf)

    # 4. Logistic Regression.
    print("\n[4/9] Logistic Regression")
    output["logistic_regression"] = _run_logistic(x_raw, y, skf)

    # 5. Weighted voting.
    print("\n[5/9] Weighted Voting")
    output["weighted_voting"] = _run_weighted_voting(df, y, models)

    # 6. Single-model ablation.
    print("\n[6/9] Single-Model Ablation")
    ablation = _run_single_ablation(
        x_scaled, y, models, best_rf_config, skf
    )
    output["single_ablation"] = ablation

    # 7. Multi-model ablation.
    print("\n[7/9] Multi-Model Ablation")
    output["multi_ablation"] = _run_multi_ablation(
        x_scaled, y, models,
        ablation["safe_to_drop"],
        best_rf_config,
        ablation["baseline_auc"],
        skf,
    )

    # 8. Threshold optimization.
    print("\n[8/9] Threshold Optimization")
    output["thresholds"] = _run_threshold_optimization(
        x_scaled, y, best_rf_config, skf
    )

    # 9. Prior-adjusted precision.
    print("\n[9/9] Prior-Adjusted Precision (Bayes)")
    output["prior_adjusted"] = _run_prior_adjusted(
        x_scaled, y, best_rf_config, skf
    )

    # Write results.
    out_path = RESULTS_DIR / f"{modality}_ensemble_rd.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
