#!/usr/bin/env python3
"""Compute per-model metrics, diversity analysis, and cross-cuts from
GPU benchmark predictions.

Consumes ``eval/results/{modality}_predictions.json`` (produced by the
benchmark runner) and writes ``eval/results/{modality}_metrics.json``
with per-model scalar metrics, per-generator AUC, era/license cross-cuts,
pairwise correlation, agreement rates, and unique-detection counts.

Usage:
    python compute_metrics.py --modality image
    python compute_metrics.py --modality audio --threshold 0.4
    python compute_metrics.py --modality video
"""

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"

# ---------------------------------------------------------------------------
# Generator era mapping
# ---------------------------------------------------------------------------

_ERA_MAP: dict[str, str] = {}

_PRE_2024 = [
    "stylegan", "stylegan2", "stylegan3", "stargan", "starganv2",
    "sd_1", "sd_1.4", "sd_1.5", "sd_2", "sd_2.0", "sd_2.1",
    "stable_diffusion_1", "stable_diffusion_2",
    "dalle_2", "dall_e_2",
    "midjourney_v4", "midjourney_v5", "midjourney_5",
    "melgan", "waveglow", "tacotron", "tacotron2",
    "faceswap", "deepfacelab", "face2face", "faceshifter",
    "progan", "biggan", "vqgan", "glide",
    "glow_tts", "hifigan", "wavenet", "wavernn",
    "asvspoof",
]

_2024 = [
    "sd_xl", "sdxl", "stable_diffusion_xl",
    "sd_3", "sd_3.0", "stable_diffusion_3",
    "dalle_3", "dall_e_3",
    "flux", "flux_1", "flux_1.0",
    "midjourney_v6", "midjourney_6",
    "sora",
    "kling", "runway", "pika",
    "bark", "elevenlabs", "suno",
    "xtts", "valle", "voicebox",
    "df40",
]

_2025_PLUS = [
    "midjourney_v7", "midjourney_7",
    "gpt_image", "gpt_image_1", "gpt_4o_image",
    "imagen_4", "imagen_4.0",
    "ideogram_3", "ideogram_3.0",
    "recraft", "recraft_v3",
    "grok_2_image", "grok_2", "grok_image",
    "veo", "veo_2",
    "minimax",
]

for _g in _PRE_2024:
    _ERA_MAP[_g] = "pre-2024"
for _g in _2024:
    _ERA_MAP[_g] = "2024"
for _g in _2025_PLUS:
    _ERA_MAP[_g] = "2025+"


def _get_era(generator: str) -> str:
    """Map a generator name to an era tag.

    Uses prefix matching so that e.g. ``asvspoof_A01`` maps to
    ``pre-2024`` via the ``asvspoof`` prefix.

    Args:
        generator: Raw generator string from the predictions file.

    Returns:
        One of ``"pre-2024"``, ``"2024"``, ``"2025+"``, or
        ``"unknown"``.
    """
    gen_lower = generator.lower().strip()
    # Direct hit.
    if gen_lower in _ERA_MAP:
        return _ERA_MAP[gen_lower]
    # Prefix match (longest first for specificity).
    for key in sorted(_ERA_MAP, key=len, reverse=True):
        if gen_lower.startswith(key):
            return _ERA_MAP[key]
    return "unknown"


# ---------------------------------------------------------------------------
# Generator license mapping
# ---------------------------------------------------------------------------

_OPEN_PREFIXES = [
    "stylegan", "stargan", "sd_", "sdxl", "stable_diffusion",
    "flux", "progan", "biggan", "vqgan", "glide",
    "melgan", "waveglow", "tacotron", "hifigan", "wavenet",
    "wavernn", "glow_tts", "bark", "xtts", "valle",
    "faceswap", "deepfacelab", "face2face", "faceshifter",
    "asvspoof", "df40",
]

_CLOSED_PREFIXES = [
    "dalle", "dall_e", "midjourney",
    "sora", "kling", "runway", "pika",
    "elevenlabs", "suno", "voicebox",
    "gpt_image", "gpt_4o", "imagen",
    "ideogram", "recraft", "grok",
    "veo", "minimax",
]


def _get_license(generator: str) -> str:
    """Map a generator name to ``"open"`` or ``"closed"``.

    Args:
        generator: Raw generator string.

    Returns:
        ``"open"``, ``"closed"``, or ``"unknown"``.
    """
    gen_lower = generator.lower().strip()
    for prefix in _OPEN_PREFIXES:
        if gen_lower.startswith(prefix):
            return "open"
    for prefix in _CLOSED_PREFIXES:
        if gen_lower.startswith(prefix):
            return "closed"
    return "unknown"


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _compute_ece(labels: np.ndarray, probs: np.ndarray,
                 n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Args:
        labels: Binary ground truth array.
        probs: Predicted probability array.
        n_bins: Number of calibration bins.

    Returns:
        ECE value in [0, 1].
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(labels)) if len(labels) > 0 else 0.0


def _bootstrap_auc(labels: np.ndarray, probs: np.ndarray,
                   n_iter: int = 1000, seed: int = 42) -> dict:
    """Bootstrap 95% CI on AUC-ROC.

    Args:
        labels: Binary ground truth.
        probs: Predicted probabilities.
        n_iter: Bootstrap iterations.
        seed: RNG seed.

    Returns:
        Dict with ``auc``, ``ci_95_lower``, ``ci_95_upper``.
    """
    point_auc = float(roc_auc_score(labels, probs))
    rng = np.random.RandomState(seed)
    n = len(labels)
    aucs = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        bl, bp = labels[idx], probs[idx]
        if len(set(bl)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(bl, bp))
        except ValueError:
            continue
    if not aucs:
        return {"auc": point_auc,
                "ci_95_lower": point_auc,
                "ci_95_upper": point_auc}
    return {
        "auc": round(point_auc, 4),
        "ci_95_lower": round(float(np.percentile(aucs, 2.5)), 4),
        "ci_95_upper": round(float(np.percentile(aucs, 97.5)), 4),
    }


def _compute_model_metrics(labels: np.ndarray, probs: np.ndarray,
                           threshold: float, n_errors: int,
                           n_total_files: int) -> dict:
    """Full metric suite for a single model.

    Args:
        labels: Binary ground truth.
        probs: Predicted probabilities.
        threshold: Classification threshold.
        n_errors: Count of error/timeout predictions for this model.
        n_total_files: Total file count including errors.

    Returns:
        Dict of all scalar metrics.
    """
    preds = (probs >= threshold).astype(int)

    # Confusion matrix.
    tn, fp, fn, tp = confusion_matrix(
        labels, preds, labels=[0, 1]
    ).ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Optimal threshold (Youden's J).
    fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs)
    j_scores = tpr_arr - fpr_arr
    best_idx = int(j_scores.argmax())
    optimal_thresh = (float(thresholds[best_idx])
                      if best_idx < len(thresholds) else 0.5)

    # EER from ROC curve.
    fnr_arr = 1 - tpr_arr
    eer_idx = int(np.nanargmin(np.abs(fpr_arr - fnr_arr)))
    eer = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2)

    result = {
        "auc_roc": round(float(roc_auc_score(labels, probs)), 4),
        "auc_pr": round(float(average_precision_score(labels, probs)), 4),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(
            precision_score(labels, preds, zero_division=0)
        ), 4),
        "recall": round(float(
            recall_score(labels, preds, zero_division=0)
        ), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "specificity": round(specificity, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "mcc": round(float(matthews_corrcoef(labels, preds)), 4),
        "optimal_threshold": round(optimal_thresh, 4),
        "eer": round(eer, 4),
        "ece": round(_compute_ece(labels, probs), 4),
        "bootstrap_ci": _bootstrap_auc(labels, probs),
        "error_rate": round(n_errors / n_total_files, 4) if n_total_files else 0.0,
        "n_valid": int(len(labels)),
        "n_errors": int(n_errors),
    }
    return result


# ---------------------------------------------------------------------------
# Per-model x per-generator AUC (global real pool)
# ---------------------------------------------------------------------------

def _per_model_per_generator(
    df: pd.DataFrame,
    labels: np.ndarray,
    generators: list[str],
    models: list[str],
) -> dict:
    """Compute per-model x per-generator AUC using global real pool.

    For each fake generator, pool ALL real samples as negatives and
    compute AUC against that generator's fakes.

    Args:
        df: DataFrame with one column per model (probability values).
        labels: Binary labels aligned with df rows.
        generators: Generator string per row.
        models: Model name list.

    Returns:
        Nested dict: model -> generator -> {auc, n_fake, low_confidence}.
    """
    real_mask = labels == 0
    result: dict[str, dict] = {}

    for model in models:
        model_result: dict[str, dict] = {}
        real_probs = df.loc[real_mask, model].values
        real_labels = np.zeros(len(real_probs), dtype=int)

        gen_series = pd.Series(generators)
        fake_mask = labels == 1
        unique_gens = gen_series[fake_mask].unique()

        for gen in sorted(unique_gens):
            gen_mask = fake_mask & (gen_series == gen)
            fake_probs = df.loc[gen_mask, model].values
            n_fake = int(len(fake_probs))
            if n_fake == 0:
                continue

            combined_labels = np.concatenate([real_labels,
                                              np.ones(n_fake, dtype=int)])
            combined_probs = np.concatenate([real_probs, fake_probs])

            try:
                auc = round(float(roc_auc_score(
                    combined_labels, combined_probs
                )), 4)
            except ValueError:
                auc = None

            model_result[gen] = {
                "auc": auc,
                "n_fake": n_fake,
                "low_confidence": n_fake < 100,
            }
        result[model] = model_result
    return result


# ---------------------------------------------------------------------------
# Per-model x per-era / per-license AUC
# ---------------------------------------------------------------------------

def _per_model_per_crosscut(
    df: pd.DataFrame,
    labels: np.ndarray,
    generators: list[str],
    models: list[str],
    tag_fn,
) -> dict:
    """AUC cross-cut by an arbitrary generator tag function.

    Args:
        df: Model probability DataFrame.
        labels: Binary labels.
        generators: Generator string per row.
        models: Model name list.
        tag_fn: Callable mapping generator string to a tag string.

    Returns:
        Nested dict: model -> tag -> {auc, n_fake}.
    """
    real_mask = labels == 0
    tags = [tag_fn(g) for g in generators]
    tag_series = pd.Series(tags)
    result: dict[str, dict] = {}

    for model in models:
        model_result: dict[str, dict] = {}
        real_probs = df.loc[real_mask, model].values
        real_labels_arr = np.zeros(len(real_probs), dtype=int)

        for tag in sorted(tag_series.unique()):
            tag_mask = (labels == 1) & (tag_series == tag)
            fake_probs = df.loc[tag_mask, model].values
            n_fake = int(len(fake_probs))
            if n_fake == 0:
                continue

            combined_labels = np.concatenate([
                real_labels_arr, np.ones(n_fake, dtype=int)
            ])
            combined_probs = np.concatenate([real_probs, fake_probs])

            try:
                auc = round(float(roc_auc_score(
                    combined_labels, combined_probs
                )), 4)
            except ValueError:
                auc = None

            model_result[tag] = {"auc": auc, "n_fake": n_fake}
        result[model] = model_result
    return result


# ---------------------------------------------------------------------------
# Correlation, agreement, unique detections
# ---------------------------------------------------------------------------

def _compute_correlation_block(
    df: pd.DataFrame,
    labels: np.ndarray,
    models: list[str],
    threshold: float,
) -> dict:
    """Pairwise Pearson correlation, agreement, and unique detections.

    Args:
        df: Model probability DataFrame (only rows where ALL models have
            valid results).
        labels: Binary labels for those rows.
        models: Model name list.
        threshold: Binary classification threshold.

    Returns:
        Dict with ``n_valid_files``, ``correlation``, ``agreement``,
        and ``unique_detections``.
    """
    n_valid = int(len(df))

    # Pairwise Pearson correlation of probability outputs.
    corr_dict: dict[str, float] = {}
    for m1, m2 in combinations(models, 2):
        key = f"{m1}_vs_{m2}"
        r = float(np.corrcoef(df[m1].values, df[m2].values)[0, 1])
        corr_dict[key] = round(r, 4)

    # Pairwise agreement rate (same binary prediction at threshold).
    binary = (df[models] >= threshold).astype(int)
    agree_dict: dict[str, float] = {}
    for m1, m2 in combinations(models, 2):
        key = f"{m1}_vs_{m2}"
        agree = float((binary[m1] == binary[m2]).mean())
        agree_dict[key] = round(agree, 4)

    # Unique detections: files where ONLY this model correctly flags a
    # fake (prob > threshold while all others <= threshold).
    unique: dict[str, int] = {}
    fake_mask = labels == 1
    for model in models:
        others = [m for m in models if m != model]
        this_correct = (df[model].values > threshold) & fake_mask
        others_wrong = np.ones(len(df), dtype=bool)
        for other in others:
            others_wrong &= (df[other].values <= threshold)
        unique[model] = int((this_correct & others_wrong).sum())

    return {
        "n_valid_files": n_valid,
        "correlation": corr_dict,
        "agreement": agree_dict,
        "unique_detections": unique,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_predictions(modality: str) -> tuple[list[dict], list[str]]:
    """Load the predictions JSON for a modality.

    Supports two formats:
    - Flat list: ``{"predictions": [...]}``.
    - Bare list: ``[{...}, ...]``.

    Args:
        modality: One of ``image``, ``audio``, ``video``.

    Returns:
        Tuple of (prediction list, list of discovered model names).

    Raises:
        SystemExit: If the file is missing or has no usable records.
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

    if not preds:
        print("ERROR: Predictions list is empty.", file=sys.stderr)
        sys.exit(1)

    # Discover model names.
    model_names: set[str] = set()
    for p in preds:
        mp = p.get("model_results") or {}
        model_names.update(mp.keys())

    return preds, sorted(model_names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse args, compute metrics, write JSON."""
    parser = argparse.ArgumentParser(
        description="Compute per-model metrics and diversity analysis."
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=["image", "audio", "video"],
        help="Modality to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary classification threshold (default 0.5).",
    )
    args = parser.parse_args()

    modality: str = args.modality
    threshold: float = args.threshold

    print(f"Loading {modality} predictions...")
    preds, models = _load_predictions(modality)
    n_files = len(preds)
    print(f"  {n_files} files, {len(models)} models: {models}")

    # ------------------------------------------------------------------
    # Build per-model arrays (labels, probs, generators).
    # ------------------------------------------------------------------
    # For per-model metrics we use all files where that model returned a
    # valid probability.  For correlation / agreement / unique-detections
    # we restrict to files where ALL models have valid results.

    generators_all = [p.get("generator", "unknown") for p in preds]

    # Per-model metrics.
    per_model: dict[str, dict] = {}
    for model in models:
        m_labels, m_probs = [], []
        n_errors = 0
        for p in preds:
            mr = (p.get("model_results") or {}).get(model)
            prob = mr.get("probability") if isinstance(mr, dict) else mr
            if prob is not None:
                lbl = p.get("label_int", 1 if p["label"] == "fake" else 0)
                m_labels.append(lbl)
                m_probs.append(prob)
            else:
                n_errors += 1

        if len(m_labels) < 2 or len(set(m_labels)) < 2:
            per_model[model] = {"error": "Not enough data",
                                "n_valid": len(m_labels),
                                "n_errors": n_errors}
            continue

        la = np.array(m_labels)
        pa = np.array(m_probs)
        per_model[model] = _compute_model_metrics(
            la, pa, threshold, n_errors, n_files
        )
        print(f"  {model}: AUC={per_model[model]['auc_roc']}")

    # ------------------------------------------------------------------
    # Build full DataFrame (only rows with all models valid) for
    # per-generator, per-era, per-license, correlation.
    # ------------------------------------------------------------------
    rows = []
    row_labels = []
    row_generators = []
    for p in preds:
        mp = p.get("model_results") or {}
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
            row_labels.append(p["label"])
            row_generators.append(p.get("generator", "unknown"))

    if not rows:
        print("WARNING: No files have valid results for ALL models.",
              file=sys.stderr)
        full_df = pd.DataFrame(columns=models)
        full_labels = np.array([], dtype=int)
        full_gens: list[str] = []
    else:
        full_df = pd.DataFrame(rows)
        full_labels = np.array(row_labels)
        full_gens = row_generators

    print(f"  {len(full_df)} files with all {len(models)} models valid")

    # Per-model x per-generator AUC.
    print("Computing per-model x per-generator AUC...")
    per_model_per_gen = _per_model_per_generator(
        full_df, full_labels, full_gens, models
    )

    # Per-model x per-era AUC.
    print("Computing per-model x per-era AUC...")
    per_model_per_era = _per_model_per_crosscut(
        full_df, full_labels, full_gens, models, _get_era
    )

    # Per-model x per-license AUC.
    print("Computing per-model x per-license AUC...")
    per_model_per_license = _per_model_per_crosscut(
        full_df, full_labels, full_gens, models, _get_license
    )

    # Correlation, agreement, unique detections.
    print("Computing correlation and agreement...")
    if len(full_df) >= 2:
        corr_block = _compute_correlation_block(
            full_df, full_labels, models, threshold
        )
    else:
        corr_block = {
            "n_valid_files": 0,
            "correlation": {},
            "agreement": {},
            "unique_detections": {},
        }

    # ------------------------------------------------------------------
    # Assemble output.
    # ------------------------------------------------------------------
    output = {
        "modality": modality,
        "n_files": n_files,
        "threshold": threshold,
        "models": models,
        "per_model": per_model,
        "per_model_per_generator": per_model_per_gen,
        "per_model_per_era": per_model_per_era,
        "per_model_per_license": per_model_per_license,
        "correlation": corr_block,
    }

    out_path = RESULTS_DIR / f"{modality}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nMetrics saved -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
