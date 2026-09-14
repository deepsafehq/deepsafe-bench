#!/usr/bin/env python3
"""Train production meta-learners for DeepSafe ensemble.

Trains modality-specific meta-learners on the full predictions dataset,
saves artifacts to models/ensemble/artifacts/, and reports CV metrics.

Each modality gets:
  - A trained classifier (.pkl)
  - A StandardScaler (.pkl)
  - A Platt calibrator (.pkl) fitted on out-of-fold predictions
  - A config JSON with model order, hyperparams, and measured AUCs

Pickle is required here because scikit-learn, XGBoost, and LightGBM
model objects cannot be serialized to JSON. The production gateway
loads these trusted local artifacts at startup.

Usage:
    python eval/train_meta_learners.py \
        --predictions eval/results/monolith_eval_*_predictions.json

    # Single modality:
    python eval/train_meta_learners.py \
        --predictions eval/results/monolith_eval_*_predictions.json \
        --modality video
"""

import argparse
import json
import logging
import pickle  # nosec B403 — sklearn/xgb/lgbm models require pickle
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Winning configs from ensemble_experiment.py sweep (2026-04-12) ──────────

IMAGE_MODELS = [
    "aide", "cospy", "effort", "fsd", "npr", "universal", "yermandy",
]
AUDIO_MODELS = ["shiftyspeech", "safeear", "nes2net"]
VIDEO_MODELS = [
    "fakestormer", "sbi", "dfd_fcg", "pwtf_dvd", "lipfd",
    "recce", "mintime", "npr_video", "univfd_video",
]

ARTIFACTS_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "ensemble" / "artifacts"
)


def _get_algo(modality: str):
    """Return the winning algorithm for a modality.

    Configs chosen from 452-experiment sweep on 15.5K samples.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        LGBMClassifier = None

    if modality == "images":
        # Winner: lgbm_200_d4, raw, all, AUC=0.9704
        if LGBMClassifier is not None:
            return LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                reg_lambda=1.0, random_state=42, verbose=-1,
            ), "LGBMClassifier"
        # Fallback: RF 500 d12 (AUC=0.9693)
        return RandomForestClassifier(
            n_estimators=500, max_depth=12, random_state=42,
        ), "RandomForestClassifier"

    if modality == "audio":
        # Winner: rf_500_d8, raw, all, AUC=0.8752
        return RandomForestClassifier(
            n_estimators=500, max_depth=8, random_state=42,
        ), "RandomForestClassifier"

    if modality == "video":
        # Winner: xgb_300_d3_reg10, raw, all, AUC=0.9031
        if XGBClassifier is not None:
            return XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                reg_lambda=10.0, use_label_encoder=False,
                eval_metric="logloss", random_state=42, verbosity=0,
            ), "XGBClassifier"
        # Fallback: GB 200 d4 (AUC=0.8886)
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42,
        ), "GradientBoostingClassifier"

    raise ValueError(f"Unknown modality: {modality}")


def load_predictions(path: str) -> pd.DataFrame:
    """Load per-file predictions JSON into a DataFrame."""
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["y"] = (df["label"] == "fake").astype(int)
    return df


def compute_optimal_threshold(y_true, y_scores):
    """Find threshold that maximizes F1."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def _compute_ece(y_true, y_probs, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_probs >= bins[i]) & (y_probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_probs[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def train_modality(
    df: pd.DataFrame,
    modality: str,
    models: list[str],
) -> dict:
    """Train a production meta-learner for one modality.

    Steps:
        1. Compute individual model AUCs
        2. 5-fold CV to estimate ensemble AUC and get OOF predictions
        3. Fit Platt calibrator on OOF predictions
        4. Retrain final model on ALL data
        5. Save artifacts
    """
    mod_df = df[df["modality"] == modality].copy()
    y = mod_df["y"].values
    n_real = (y == 0).sum()
    n_fake = (y == 1).sum()

    avail = [m for m in models if f"prob_{m}" in mod_df.columns]
    prob_cols = [f"prob_{m}" for m in avail]
    X = mod_df[prob_cols].fillna(0.5).values

    logger.info(
        "=== %s: %d samples (%d real, %d fake), %d models ===",
        modality.upper(), len(mod_df), n_real, n_fake, len(avail),
    )

    # ── Individual model AUCs ───────────────────────────────────────────
    model_aucs = {}
    for m in avail:
        col = f"prob_{m}"
        probs = mod_df[col].fillna(0.5).values
        try:
            auc = roc_auc_score(y, probs)
        except ValueError:
            auc = 0.5
        model_aucs[m] = round(auc, 4)
        logger.info("  %s: AUC=%.4f", m, auc)

    # ── Get algorithm ───────────────────────────────────────────────────
    algo, algo_name = _get_algo(modality)
    logger.info("  Algorithm: %s", algo_name)

    # ── Fit scaler on all data ──────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 5-fold CV for metrics and OOF predictions ───────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(
        algo, X_scaled, y, cv=cv, method="predict_proba",
    )[:, 1]

    cv_auc = roc_auc_score(y, oof_probs)
    cv_f1 = f1_score(y, (oof_probs >= 0.5).astype(int), zero_division=0)
    opt_t, opt_f1 = compute_optimal_threshold(y, oof_probs)
    ece = _compute_ece(y, oof_probs)

    logger.info(
        "  CV AUC=%.4f, F1=%.4f, opt_F1=%.4f (t=%.3f), ECE=%.4f",
        cv_auc, cv_f1, opt_f1, opt_t, ece,
    )

    # ── Platt calibration on OOF predictions ────────────────────────────
    calibrator = LogisticRegression(C=1.0, max_iter=1000)
    calibrator.fit(oof_probs.reshape(-1, 1), y)
    cal_probs = calibrator.predict_proba(oof_probs.reshape(-1, 1))[:, 1]
    cal_auc = roc_auc_score(y, cal_probs)
    cal_ece = _compute_ece(y, cal_probs)
    cal_opt_t, cal_opt_f1 = compute_optimal_threshold(y, cal_probs)
    logger.info(
        "  Post-calibration: AUC=%.4f, ECE=%.4f, opt_F1=%.4f (t=%.3f)",
        cal_auc, cal_ece, cal_opt_f1, cal_opt_t,
    )

    # ── Train final model on ALL data ───────────────────────────────────
    final_algo, _ = _get_algo(modality)
    final_algo.fit(X_scaled, y)
    logger.info("  Final model trained on %d samples", len(y))

    # ── Verify final model on training data (sanity check) ──────────────
    train_probs = final_algo.predict_proba(X_scaled)[:, 1]
    train_auc = roc_auc_score(y, train_probs)
    logger.info("  Training AUC (sanity check)=%.4f", train_auc)

    # ── Save artifacts ──────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = modality.rstrip("s") if modality == "images" else modality

    model_path = ARTIFACTS_DIR / f"{prefix}_meta_learner.pkl"
    scaler_path = ARTIFACTS_DIR / f"{prefix}_scaler.pkl"
    calibrator_path = ARTIFACTS_DIR / f"{prefix}_calibrator.pkl"
    config_path = ARTIFACTS_DIR / f"{prefix}_config.json"

    # nosec B301 — pickle required for sklearn/xgb/lgbm model objects.
    # These are trusted local artifacts written by this training script
    # and loaded only by the DeepSafe gateway at startup.
    with open(model_path, "wb") as f:
        pickle.dump(final_algo, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)

    config = {
        "model_order": avail,
        "algorithm": algo_name,
        "hyperparams": _extract_hyperparams(final_algo),
        "cv_auc": round(cv_auc, 4),
        "cv_f1_at_05": round(cv_f1, 4),
        "optimal_threshold": round(opt_t, 3),
        "optimal_f1": round(opt_f1, 4),
        "ece_before_calibration": round(ece, 4),
        "ece_after_calibration": round(cal_ece, 4),
        "calibrated_auc": round(cal_auc, 4),
        "calibrated_optimal_threshold": round(cal_opt_t, 3),
        "calibrated_optimal_f1": round(cal_opt_f1, 4),
        "calibrator_saved": True,
        "n_samples": len(y),
        "n_real": int(n_real),
        "n_fake": int(n_fake),
        "measured_model_aucs": model_aucs,
        "training_date": str(date.today()),
        "training_data": "monolith_eval_medium_15499_patched",
        "note": (
            f"{len(avail)}-model {algo_name} trained on 15,499-sample "
            f"medium eval. Replaces small-eval (50 sample) artifacts."
        ),
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("  Artifacts saved to %s/", ARTIFACTS_DIR)
    logger.info("    %s", model_path.name)
    logger.info("    %s", scaler_path.name)
    logger.info("    %s", calibrator_path.name)
    logger.info("    %s", config_path.name)

    return config


def _extract_hyperparams(algo) -> dict:
    """Pull key hyperparams from a fitted estimator."""
    params = algo.get_params()
    keys = [
        "n_estimators", "max_depth", "learning_rate", "reg_lambda",
        "reg_alpha", "min_child_weight", "subsample", "colsample_bytree",
        "C", "penalty", "random_state",
    ]
    return {k: v for k, v in params.items() if k in keys and v is not None}


def main():
    parser = argparse.ArgumentParser(
        description="Train production meta-learners for DeepSafe ensemble",
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to predictions JSON",
    )
    parser.add_argument(
        "--modality", default=None,
        choices=["images", "audio", "video"],
        help="Train only one modality (default: all)",
    )
    args = parser.parse_args()

    df = load_predictions(args.predictions)
    logger.info("Loaded %d predictions", len(df))

    modalities = [args.modality] if args.modality else ["images", "audio",
                                                         "video"]
    configs = {}
    for modality in modalities:
        models = {
            "images": IMAGE_MODELS,
            "audio": AUDIO_MODELS,
            "video": VIDEO_MODELS,
        }[modality]
        config = train_modality(df, modality, models)
        configs[modality] = config

    # ── Final summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PRODUCTION META-LEARNER TRAINING COMPLETE")
    logger.info("=" * 70)
    for mod, cfg in configs.items():
        logger.info(
            "\n  %s: %s (%d models, %d samples)",
            mod.upper(), cfg["algorithm"],
            len(cfg["model_order"]), cfg["n_samples"],
        )
        logger.info(
            "    CV AUC=%.4f  F1=%.4f  ECE=%.4f (calibrated=%.4f)",
            cfg["cv_auc"], cfg["cv_f1_at_05"],
            cfg["ece_before_calibration"], cfg["ece_after_calibration"],
        )
        logger.info(
            "    Optimal: F1=%.4f at threshold=%.3f",
            cfg["optimal_f1"], cfg["optimal_threshold"],
        )
        logger.info(
            "    Models: %s", cfg["model_order"],
        )
    logger.info("\nArtifacts saved to: %s/", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
