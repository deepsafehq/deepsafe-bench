#!/usr/bin/env python3
"""Compute all metrics from an existing per-file prediction JSON.

Loads a prediction file (eval/results/archive/eval_per_file_predictions.json
or similar), computes ensemble-level metrics, per-model metrics, and
per-generator AUC for each modality, then prints a summary and saves a
timestamped JSON report.

Usage:
    uv run python -m eval.run_metrics_only
    uv run python -m eval.run_metrics_only --predictions path/to/preds.json
    uv run python -m eval.run_metrics_only --output path/to/report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from deepsafe_eval.config import RESULTS_DIR, get_family
from deepsafe_eval.metrics import (
    bootstrap_auc_ci,
    compute_all_metrics,
    compute_per_generator_auc,
)

_DEFAULT_PREDICTIONS = (
    Path(__file__).parent / "results" / "archive" / "eval_per_file_predictions.json"
)

# Modality keys expected in the prediction JSON.
_MODALITIES = ("images", "audio", "video")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute all metrics from an existing prediction JSON."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_DEFAULT_PREDICTIONS,
        help="Path to the per-file prediction JSON (default: archive copy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the metrics JSON. "
            "Default: eval/results/YYYY-MM-DD-HH-MM-metrics.json"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        choices=["images", "audio", "video"],
        help="Evaluate only this modality (default: all).",
    )
    return parser.parse_args()


def _load_predictions(path: Path) -> dict:
    """Load and validate the prediction JSON.

    Args:
        path: Filesystem path to the prediction JSON.

    Returns:
        Dict with modality keys ('images', 'audio', 'video'),
        each containing a list of prediction dicts.

    Raises:
        SystemExit: If the file cannot be loaded or has an invalid format.
    """
    if not path.exists():
        print(f"ERROR: Predictions file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        print("ERROR: Top-level JSON must be an object.", file=sys.stderr)
        sys.exit(1)
    return data


def _compute_modality_metrics(
    predictions: list[dict],
    threshold: float,
) -> dict:
    """Compute all metrics for a single modality.

    Args:
        predictions: List of prediction dicts, each with keys:
            'label' (0/1), 'ensemble_prob' (float), 'generator' (str),
            'model_probs' (dict[str, float]).
        threshold: Classification threshold.

    Returns:
        Dict with 'ensemble', 'per_model', and 'per_generator' sub-dicts.
    """
    result = {}

    # --- Ensemble metrics ---
    ensemble_labels = []
    ensemble_probs = []
    for p in predictions:
        if p.get("ensemble_prob") is not None:
            ensemble_labels.append(p["label"])
            ensemble_probs.append(p["ensemble_prob"])

    if len(ensemble_labels) >= 2 and len(set(ensemble_labels)) == 2:
        result["ensemble"] = compute_all_metrics(
            ensemble_labels, ensemble_probs, threshold
        )
    else:
        result["ensemble"] = {"error": "Not enough data for metrics."}

    # --- Per-model metrics ---
    # Discover all model names across predictions.
    model_names = set()
    for p in predictions:
        if "model_probs" in p and p["model_probs"]:
            model_names.update(p["model_probs"].keys())

    per_model = {}
    for model in sorted(model_names):
        labels = []
        probs = []
        for p in predictions:
            prob = (p.get("model_probs") or {}).get(model)
            if prob is not None:
                labels.append(p["label"])
                probs.append(prob)
        if len(labels) >= 2 and len(set(labels)) == 2:
            per_model[model] = compute_all_metrics(labels, probs, threshold)
        else:
            per_model[model] = {
                "error": "Not enough data.",
                "n_total": len(labels),
            }
    result["per_model"] = per_model

    # --- Per-generator AUC (ensemble) ---
    result["per_generator_auc"] = compute_per_generator_auc(predictions)

    # --- Per-generator-family AUC ---
    family_map: dict[str, list[float]] = {}
    real_probs_all = [
        p["ensemble_prob"]
        for p in predictions
        if p["label"] == 0 and p.get("ensemble_prob") is not None
    ]
    for p in predictions:
        if p["label"] == 1 and p.get("ensemble_prob") is not None:
            fam = get_family(p["generator"])
            family_map.setdefault(fam, []).append(p["ensemble_prob"])

    per_family = {}
    real_labels_all = [0] * len(real_probs_all)
    for fam, fake_probs in family_map.items():
        combined_labels = real_labels_all + [1] * len(fake_probs)
        combined_probs = real_probs_all + fake_probs
        if len(set(combined_labels)) < 2:
            continue
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(combined_labels, combined_probs))
            per_family[fam] = round(auc, 4)
        except ValueError:
            per_family[fam] = None
    result["per_family_auc"] = per_family

    return result


def _print_modality_report(modality: str, metrics: dict) -> None:
    """Print a human-readable summary for one modality.

    Args:
        modality: Modality name (images, audio, video).
        metrics: The metrics dict from _compute_modality_metrics.
    """
    print(f"\n{'='*70}")
    print(f"  {modality.upper()}")
    print(f"{'='*70}")

    # Ensemble
    ens = metrics.get("ensemble", {})
    if "error" not in ens:
        print(f"\n  Ensemble:")
        print(
            f"    AUC={ens['auc_roc']:.4f}  "
            f"AP={ens['auc_pr']:.4f}  "
            f"Acc={ens['accuracy']:.4f}  "
            f"F1={ens['f1']:.4f}  "
            f"EER={ens['eer']:.4f}"
        )
        ci = ens.get("bootstrap_ci", {})
        if ci:
            print(
                f"    AUC 95% CI: [{ci['ci_95_lower']:.4f}, "
                f"{ci['ci_95_upper']:.4f}]"
            )
        print(
            f"    TP={ens['tp']}  TN={ens['tn']}  "
            f"FP={ens['fp']}  FN={ens['fn']}  "
            f"(N={ens['n_total']})"
        )
        print(
            f"    Optimal Threshold: {ens['optimal_threshold']:.4f}  "
            f"ECE: {ens['ece']:.4f}"
        )
    else:
        print(f"\n  Ensemble: {ens['error']}")

    # Per-model table
    pm = metrics.get("per_model", {})
    if pm:
        print(f"\n  Per-Model:")
        print(
            f"  {'Model':<20} {'AUC':>7} {'AP':>7} {'Acc':>7} "
            f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'EER':>7} {'N':>6}"
        )
        print(f"  {'-'*78}")
        for name in sorted(pm.keys()):
            m = pm[name]
            if "error" in m:
                print(f"  {name:<20} {'---':>7}  ({m['error']})")
                continue
            print(
                f"  {name:<20} "
                f"{m['auc_roc']:>7.4f} {m['auc_pr']:>7.4f} "
                f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} "
                f"{m['recall']:>7.4f} {m['f1']:>7.4f} "
                f"{m['eer']:>7.4f} {m['n_total']:>6}"
            )

    # Per-generator AUC
    gen_auc = metrics.get("per_generator_auc", {})
    if gen_auc:
        print(f"\n  Per-Generator AUC (ensemble):")
        for gen in sorted(gen_auc.keys()):
            auc_val = gen_auc[gen]
            bar = ""
            if auc_val is not None:
                filled = int(auc_val * 20)
                bar = "#" * filled + "." * (20 - filled)
            print(
                f"    {gen:<30} "
                f"{auc_val if auc_val is not None else 'N/A':>7}  "
                f"[{bar}]"
            )

    # Per-family AUC
    fam_auc = metrics.get("per_family_auc", {})
    if fam_auc:
        print(f"\n  Per-Family AUC (ensemble):")
        for fam in sorted(fam_auc.keys()):
            auc_val = fam_auc[fam]
            print(f"    {fam:<30} " f"{auc_val if auc_val is not None else 'N/A':>7}")


def main() -> None:
    """Run the metrics-only evaluation pipeline."""
    args = _parse_args()

    print("=" * 70)
    print("  DeepSafe Metrics Evaluator (from existing predictions)")
    print(f"  Predictions: {args.predictions}")
    print(f"  Threshold:   {args.threshold}")
    print("=" * 70)

    data = _load_predictions(args.predictions)

    modalities = [args.modality] if args.modality else _MODALITIES
    report = {
        "metadata": {
            "predictions_file": str(args.predictions),
            "threshold": args.threshold,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    for modality in modalities:
        predictions = data.get(modality, [])
        if not predictions:
            print(f"\n  [{modality}] No predictions found -- skipping.")
            continue

        print(f"\n  [{modality}] Computing metrics for {len(predictions)} samples...")
        metrics = _compute_modality_metrics(predictions, args.threshold)
        report[modality] = metrics
        _print_modality_report(modality, metrics)

    # Save JSON report
    if args.output:
        out_path = args.output
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d-%H-%M")
        out_path = RESULTS_DIR / f"{stamp}-metrics.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Metrics saved -> {out_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
