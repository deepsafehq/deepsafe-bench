#!/usr/bin/env python3
"""Exhaustive meta-learner experiment for DeepSafe ensemble.

Tries hundreds of algorithm/feature/hyperparameter combinations per modality
to find the best ensemble strategy. Outputs a ranked leaderboard.

Usage:
    python eval/ensemble_experiment.py --predictions eval/results/monolith_eval_*_predictions.json
"""

import argparse
import json
import itertools
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Model sets per modality ──────────────────────────────────────────────────

IMAGE_MODELS = [
    "aide", "cospy", "effort", "fsd", "npr", "universal", "yermandy",
]
AUDIO_MODELS = ["shiftyspeech", "safeear", "nes2net"]
VIDEO_MODELS = [
    "fakestormer", "sbi", "dfd_fcg", "pwtf_dvd", "lipfd",
    "recce", "mintime", "npr_video", "univfd_video",
]
PROVENANCE_MODELS = ["c2pa", "sdxl_watermark", "audioseal", "videoseal"]

MODALITY_MODELS = {
    "images": IMAGE_MODELS,
    "audio": AUDIO_MODELS,
    "video": VIDEO_MODELS,
}


def load_predictions(path: str) -> pd.DataFrame:
    """Load per-file predictions JSON into a DataFrame."""
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["y"] = (df["label"] == "fake").astype(int)
    return df


def get_feature_matrix(
    df: pd.DataFrame,
    models: list[str],
    feature_mode: str = "raw",
) -> np.ndarray:
    """Build feature matrix from model probabilities.

    Feature modes:
        raw: Raw probabilities [0, 1]
        logodds: Log-odds transform log(p / (1-p))
        augmented: Raw + pairwise diffs + variance + max-min
        rank: Rank of each model's probability within the sample
        topk: Only models with individual AUC > 0.55
    """
    prob_cols = [f"prob_{m}" for m in models]
    X = df[prob_cols].fillna(0.5).values

    if feature_mode == "raw":
        return X

    if feature_mode == "logodds":
        eps = 1e-6
        X_clip = np.clip(X, eps, 1 - eps)
        return np.log(X_clip / (1 - X_clip))

    if feature_mode == "augmented":
        eps = 1e-6
        X_clip = np.clip(X, eps, 1 - eps)
        logodds = np.log(X_clip / (1 - X_clip))
        variance = np.var(X, axis=1, keepdims=True)
        spread = (np.max(X, axis=1) - np.min(X, axis=1)).reshape(-1, 1)
        mean_prob = np.mean(X, axis=1, keepdims=True)
        max_prob = np.max(X, axis=1, keepdims=True)
        # Top-2 agreement: difference between 2 highest probs
        sorted_X = np.sort(X, axis=1)
        top2_diff = (sorted_X[:, -1] - sorted_X[:, -2]).reshape(-1, 1)
        return np.hstack([X, logodds, variance, spread, mean_prob,
                          max_prob, top2_diff])

    if feature_mode == "rank":
        from scipy.stats import rankdata
        return np.apply_along_axis(
            lambda row: rankdata(row) / len(row), 1, X,
        )

    return X


def get_topk_models(
    df: pd.DataFrame,
    models: list[str],
    min_auc: float = 0.55,
) -> list[str]:
    """Return models with individual AUC above threshold."""
    y = df["y"].values
    selected = []
    for m in models:
        col = f"prob_{m}"
        if col not in df.columns:
            continue
        probs = df[col].fillna(0.5).values
        try:
            auc = roc_auc_score(y, probs)
            if auc >= min_auc:
                selected.append(m)
        except ValueError:
            pass
    return selected if selected else models[:3]  # fallback to top 3


def compute_optimal_threshold(y_true, y_scores):
    """Find threshold that maximizes F1."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def build_experiments():
    """Generate all experiment configurations."""
    experiments = []

    # ── Algorithm definitions ────────────────────────────────────────────
    algos = {
        # Linear models
        "lr_l2_c001": LogisticRegression(C=0.01, penalty="l2", max_iter=2000,
                                         solver="lbfgs"),
        "lr_l2_c01": LogisticRegression(C=0.1, penalty="l2", max_iter=2000,
                                        solver="lbfgs"),
        "lr_l2_c1": LogisticRegression(C=1.0, penalty="l2", max_iter=2000,
                                       solver="lbfgs"),
        "lr_l2_c10": LogisticRegression(C=10.0, penalty="l2", max_iter=2000,
                                        solver="lbfgs"),
        "lr_l1_c01": LogisticRegression(C=0.1, penalty="l1", max_iter=2000,
                                        solver="saga"),
        "lr_l1_c1": LogisticRegression(C=1.0, penalty="l1", max_iter=2000,
                                       solver="saga"),
        "lr_enet_c1": LogisticRegression(C=1.0, penalty="elasticnet",
                                         l1_ratio=0.5, max_iter=2000,
                                         solver="saga"),
        "lr_enet_c01": LogisticRegression(C=0.1, penalty="elasticnet",
                                          l1_ratio=0.5, max_iter=2000,
                                          solver="saga"),
        # Tree models
        "rf_100_d4": RandomForestClassifier(n_estimators=100, max_depth=4,
                                            random_state=42),
        "rf_100_d6": RandomForestClassifier(n_estimators=100, max_depth=6,
                                            random_state=42),
        "rf_300_d6": RandomForestClassifier(n_estimators=300, max_depth=6,
                                            random_state=42),
        "rf_500_d8": RandomForestClassifier(n_estimators=500, max_depth=8,
                                            random_state=42),
        "rf_500_d12": RandomForestClassifier(n_estimators=500, max_depth=12,
                                             random_state=42),
        "et_100_d4": ExtraTreesClassifier(n_estimators=100, max_depth=4,
                                          random_state=42),
        "et_300_d6": ExtraTreesClassifier(n_estimators=300, max_depth=6,
                                          random_state=42),
        "gb_100_d3": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                learning_rate=0.1,
                                                random_state=42),
        "gb_200_d4": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                learning_rate=0.05,
                                                random_state=42),
        "gb_300_d3_lr01": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.01,
            random_state=42,
        ),
        "ada_100": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42,
        ),
        "ada_200": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.05, random_state=42,
        ),
        "dt_d3": DecisionTreeClassifier(max_depth=3, random_state=42),
        "dt_d5": DecisionTreeClassifier(max_depth=5, random_state=42),
        "bag_lr": BaggingClassifier(
            estimator=LogisticRegression(C=1.0, max_iter=2000),
            n_estimators=20, random_state=42,
        ),
        # SVM — linear only (RBF is O(n^2), too slow for 10K+ samples)
        "svm_linear_c1": SVC(kernel="linear", C=1.0, probability=True,
                             random_state=42),
        "svm_linear_c01": SVC(kernel="linear", C=0.1, probability=True,
                              random_state=42),
        # KNN
        "knn_3": KNeighborsClassifier(n_neighbors=3),
        "knn_5": KNeighborsClassifier(n_neighbors=5),
        "knn_7": KNeighborsClassifier(n_neighbors=7),
        "knn_11": KNeighborsClassifier(n_neighbors=11),
        # Neural network
        "mlp_small": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000,
                                   random_state=42),
        "mlp_medium": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000,
                                    random_state=42),
        "mlp_large": MLPClassifier(hidden_layer_sizes=(64, 32),
                                   max_iter=500, random_state=42),
        # Naive Bayes
        "gnb": GaussianNB(),
        # SGD
        "sgd_log": SGDClassifier(loss="log_loss", penalty="l2",
                                 alpha=0.01, max_iter=2000,
                                 random_state=42),
        "sgd_log_l1": SGDClassifier(loss="log_loss", penalty="l1",
                                    alpha=0.01, max_iter=2000,
                                    random_state=42),
    }

    # Try XGBoost/LightGBM if available
    try:
        from xgboost import XGBClassifier
        algos.update({
            "xgb_100_d3": XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                reg_lambda=1.0, use_label_encoder=False,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
            "xgb_200_d4": XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                reg_lambda=1.0, use_label_encoder=False,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
            "xgb_300_d3_reg10": XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                reg_lambda=10.0, use_label_encoder=False,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
            "xgb_100_d2_reg5": XGBClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.1,
                reg_lambda=5.0, use_label_encoder=False,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
        })
    except ImportError:
        logger.info("XGBoost not available, skipping.")

    try:
        from lightgbm import LGBMClassifier
        algos.update({
            "lgbm_100_d3": LGBMClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                reg_lambda=1.0, random_state=42, verbose=-1,
            ),
            "lgbm_200_d4": LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                reg_lambda=1.0, random_state=42, verbose=-1,
            ),
            "lgbm_300_d3_reg10": LGBMClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.01,
                reg_lambda=10.0, random_state=42, verbose=-1,
            ),
        })
    except ImportError:
        logger.info("LightGBM not available, skipping.")

    # ── Feature modes ────────────────────────────────────────────────────
    feature_modes = ["raw", "logodds", "augmented", "rank"]

    # ── Model subsets ────────────────────────────────────────────────────
    model_subsets = ["all", "topk"]

    # ── Smart experiment generation ──────────────────────────────────────
    # Avoid redundant combos: tree methods don't benefit from scaling,
    # SVM RBF is too slow on 10K samples, augmented features only for
    # linear models where they help.
    _TREE_ALGOS = {
        "rf_", "et_", "gb_", "ada_", "dt_", "xgb_", "lgbm_", "bag_",
    }
    _SLOW_ALGOS = {"svm_rbf_c1", "svm_rbf_c10", "mlp_large", "mlp_wide"}

    for algo_name, algo in algos.items():
        is_tree = any(algo_name.startswith(p) for p in _TREE_ALGOS)

        for feat_mode in feature_modes:
            # Augmented features only for linear/SVM — trees don't
            # benefit and it doubles feature count
            if feat_mode == "augmented" and is_tree:
                continue

            for subset in model_subsets:
                # Scaling: trees don't need it, linear models do
                if is_tree:
                    scale_opts = [False]
                else:
                    scale_opts = [True, False]

                for scale in scale_opts:
                    experiments.append({
                        "algo_name": algo_name,
                        "algo": algo,
                        "feature_mode": feat_mode,
                        "model_subset": subset,
                        "scale": scale,
                    })

    return experiments


def evaluate_experiment(
    exp: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv,
) -> dict:
    """Run one experiment with cross-validation."""
    algo = exp["algo"]

    # Build pipeline
    steps = []
    if exp["scale"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", algo))
    pipe = Pipeline(steps)

    try:
        # Get cross-validated probability predictions
        y_probs = cross_val_predict(
            pipe, X, y, cv=cv, method="predict_proba",
        )[:, 1]
    except Exception:
        try:
            # Some classifiers don't support predict_proba
            y_preds = cross_val_predict(pipe, X, y, cv=cv, method="predict")
            y_probs = y_preds.astype(float)
        except Exception as e:
            return {"error": str(e)}

    y_pred_05 = (y_probs >= 0.5).astype(int)

    # Metrics at 0.5 threshold
    try:
        auc = roc_auc_score(y, y_probs)
    except ValueError:
        auc = 0.5

    f1 = f1_score(y, y_pred_05, zero_division=0)
    precision = precision_score(y, y_pred_05, zero_division=0)
    recall = recall_score(y, y_pred_05, zero_division=0)
    accuracy = accuracy_score(y, y_pred_05)

    # Optimal threshold
    opt_t, opt_f1 = compute_optimal_threshold(y, y_probs)
    y_pred_opt = (y_probs >= opt_t).astype(int)
    opt_precision = precision_score(y, y_pred_opt, zero_division=0)
    opt_recall = recall_score(y, y_pred_opt, zero_division=0)
    opt_accuracy = accuracy_score(y, y_pred_opt)

    # Calibration: Expected Calibration Error
    ece = _compute_ece(y, y_probs, n_bins=10)

    return {
        "auc": round(auc, 5),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "ece": round(ece, 4),
        "opt_threshold": round(opt_t, 3),
        "opt_f1": round(opt_f1, 4),
        "opt_precision": round(opt_precision, 4),
        "opt_recall": round(opt_recall, 4),
        "opt_accuracy": round(opt_accuracy, 4),
    }


def _compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def run_simple_baselines(df, models, y):
    """Run simple non-ML baselines for comparison."""
    prob_cols = [f"prob_{m}" for m in models if f"prob_{m}" in df.columns]
    X = df[prob_cols].fillna(0.5).values

    results = []

    # Simple average
    avg = X.mean(axis=1)
    auc = roc_auc_score(y, avg)
    f1 = f1_score(y, (avg >= 0.5).astype(int), zero_division=0)
    opt_t, opt_f1 = compute_optimal_threshold(y, avg)
    results.append({
        "name": "simple_average",
        "auc": round(auc, 5), "f1": round(f1, 4),
        "opt_threshold": round(opt_t, 3), "opt_f1": round(opt_f1, 4),
    })

    # Max (any model says fake → fake)
    mx = X.max(axis=1)
    auc = roc_auc_score(y, mx)
    f1 = f1_score(y, (mx >= 0.5).astype(int), zero_division=0)
    opt_t, opt_f1 = compute_optimal_threshold(y, mx)
    results.append({
        "name": "max_score",
        "auc": round(auc, 5), "f1": round(f1, 4),
        "opt_threshold": round(opt_t, 3), "opt_f1": round(opt_f1, 4),
    })

    # Median
    med = np.median(X, axis=1)
    auc = roc_auc_score(y, med)
    f1 = f1_score(y, (med >= 0.5).astype(int), zero_division=0)
    opt_t, opt_f1 = compute_optimal_threshold(y, med)
    results.append({
        "name": "median_score",
        "auc": round(auc, 5), "f1": round(f1, 4),
        "opt_threshold": round(opt_t, 3), "opt_f1": round(opt_f1, 4),
    })

    # AUC-weighted average
    aucs = {}
    for m in models:
        col = f"prob_{m}"
        if col not in df.columns:
            continue
        try:
            aucs[m] = roc_auc_score(y, df[col].fillna(0.5).values)
        except ValueError:
            aucs[m] = 0.5

    weights = np.array([max(aucs.get(m, 0.5) - 0.5, 0.01)
                        for m in models if f"prob_{m}" in df.columns])
    weights = weights / weights.sum()
    weighted = (X * weights).sum(axis=1)
    auc = roc_auc_score(y, weighted)
    f1 = f1_score(y, (weighted >= 0.5).astype(int), zero_division=0)
    opt_t, opt_f1 = compute_optimal_threshold(y, weighted)
    results.append({
        "name": "auc_weighted_avg",
        "auc": round(auc, 5), "f1": round(f1, 4),
        "opt_threshold": round(opt_t, 3), "opt_f1": round(opt_f1, 4),
    })

    # Best single model
    for m in models:
        col = f"prob_{m}"
        if col not in df.columns:
            continue
        probs = df[col].fillna(0.5).values
        try:
            auc = roc_auc_score(y, probs)
        except ValueError:
            auc = 0.5
        f1 = f1_score(y, (probs >= 0.5).astype(int), zero_division=0)
        opt_t, opt_f1 = compute_optimal_threshold(y, probs)
        results.append({
            "name": f"single_{m}",
            "auc": round(auc, 5), "f1": round(f1, 4),
            "opt_threshold": round(opt_t, 3), "opt_f1": round(opt_f1, 4),
        })

    return results


def run_modality_experiments(
    df: pd.DataFrame,
    modality: str,
    models: list[str],
) -> list[dict]:
    """Run all experiments for a single modality."""
    mod_df = df[df["modality"] == modality].copy()
    if len(mod_df) < 20:
        logger.warning("Skipping %s: only %d samples", modality, len(mod_df))
        return []

    y = mod_df["y"].values
    n_real = (y == 0).sum()
    n_fake = (y == 1).sum()
    logger.info(
        "=== %s: %d samples (%d real, %d fake), %d models ===",
        modality.upper(), len(mod_df), n_real, n_fake, len(models),
    )

    # Available models (have predictions)
    avail = [m for m in models if f"prob_{m}" in mod_df.columns]
    topk = get_topk_models(mod_df, avail)
    logger.info("  Available models: %s", avail)
    logger.info("  Top-K models (AUC>0.55): %s", topk)

    # Individual model AUCs
    for m in avail:
        col = f"prob_{m}"
        probs = mod_df[col].fillna(0.5).values
        try:
            auc = roc_auc_score(y, probs)
            logger.info("    %s: AUC=%.4f", m, auc)
        except ValueError:
            pass

    # Run baselines
    baselines = run_simple_baselines(mod_df, avail, y)
    logger.info("  Baselines:")
    for b in baselines:
        logger.info("    %s: AUC=%.4f, F1=%.4f, opt_F1=%.4f (t=%.3f)",
                     b["name"], b["auc"], b["f1"], b["opt_f1"],
                     b["opt_threshold"])

    # CV strategy — adapt to dataset size
    n = len(mod_df)
    if n < 100:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                     random_state=42)
    elif n < 500:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                     random_state=42)
    elif n < 5000:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        # Large datasets: 3-fold is sufficient and 40% faster
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    experiments = build_experiments()
    results = []
    total = len(experiments)

    for i, exp in enumerate(experiments):
        if (i + 1) % 50 == 0:
            logger.info("  [%d/%d] experiments completed...", i + 1, total)

        # Select model subset
        if exp["model_subset"] == "topk":
            models_to_use = topk
        else:
            models_to_use = avail

        # Build feature matrix
        X = get_feature_matrix(mod_df, models_to_use, exp["feature_mode"])

        # Skip if features > samples (overfit guaranteed)
        if X.shape[1] > X.shape[0] * 0.8:
            continue

        metrics = evaluate_experiment(exp, X, y, cv)
        if "error" in metrics:
            continue

        result = {
            "modality": modality,
            "algo": exp["algo_name"],
            "features": exp["feature_mode"],
            "model_subset": exp["model_subset"],
            "n_models": len(models_to_use),
            "n_features": X.shape[1],
            "scaled": exp["scale"],
            **metrics,
        }
        results.append(result)

    # Sort by AUC
    results.sort(key=lambda r: r["auc"], reverse=True)

    # Print top 20
    logger.info("\n  TOP 20 EXPERIMENTS (by AUC) — %s:", modality.upper())
    logger.info(
        "  %-25s %-10s %-6s %-6s  AUC     F1     P      R      "
        "ECE    optF1  optT",
        "Algorithm", "Features", "Sub", "Scale",
    )
    logger.info("  " + "-" * 110)
    for r in results[:20]:
        logger.info(
            "  %-25s %-10s %-6s %-6s  %.4f  %.4f  %.4f  %.4f  "
            "%.4f  %.4f  %.3f",
            r["algo"], r["features"], r["model_subset"],
            "Y" if r["scaled"] else "N",
            r["auc"], r["f1"], r["precision"], r["recall"],
            r["ece"], r["opt_f1"], r["opt_threshold"],
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions JSON")
    parser.add_argument("--output", default=None,
                        help="Output JSON path for results")
    parser.add_argument("--modality", default=None,
                        choices=["images", "audio", "video"],
                        help="Run only one modality")
    args = parser.parse_args()

    df = load_predictions(args.predictions)
    logger.info("Loaded %d predictions", len(df))

    all_results = {}

    modalities = [args.modality] if args.modality else ["images", "audio",
                                                         "video"]
    for modality in modalities:
        models = MODALITY_MODELS[modality]
        results = run_modality_experiments(df, modality, models)
        all_results[modality] = results
        logger.info("  %s: %d valid experiments", modality, len(results))

    # Save results
    output = args.output or str(
        Path(args.predictions).parent / "ensemble_experiments.json"
    )
    # Flatten for JSON
    flat = []
    for mod, results in all_results.items():
        flat.extend(results)

    with open(output, "w") as f:
        json.dump(flat, f, indent=2)
    logger.info("\nAll results saved: %s (%d experiments)", output, len(flat))

    # Print final summary — best per modality
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY — BEST CONFIGURATION PER MODALITY")
    logger.info("=" * 80)
    for mod in modalities:
        results = all_results.get(mod, [])
        if not results:
            continue
        best = results[0]
        logger.info(
            "\n  %s: %s (%s features, %s models, scaled=%s)",
            mod.upper(), best["algo"], best["features"],
            best["model_subset"], best["scaled"],
        )
        logger.info(
            "    AUC=%.4f  F1=%.4f  P=%.4f  R=%.4f  ECE=%.4f",
            best["auc"], best["f1"], best["precision"],
            best["recall"], best["ece"],
        )
        logger.info(
            "    Optimal: F1=%.4f at threshold=%.3f (P=%.4f R=%.4f)",
            best["opt_f1"], best["opt_threshold"],
            best["opt_precision"], best["opt_recall"],
        )


if __name__ == "__main__":
    main()
