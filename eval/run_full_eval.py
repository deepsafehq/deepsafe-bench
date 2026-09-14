#!/usr/bin/env python3
"""Complete evaluation pipeline: load predictions, compute metrics, generate report.

Chains the full DeepSafe evaluation pipeline:
1. Load per-file predictions from JSON (or run fresh inference)
2. Compute all metrics via run_metrics_only logic
3. Generate a Markdown report via deepsafe_eval.report
4. Save both JSON results and Markdown report to eval/results/

Usage:
    uv run python -m eval.run_full_eval
    uv run python -m eval.run_full_eval --predictions path/to/preds.json
    uv run python -m eval.run_full_eval --output-dir eval/results/custom
"""

import argparse
import json
import sys
import time
from pathlib import Path

from deepsafe_eval.config import DATASET_ROOT, RESULTS_DIR
from deepsafe_eval.report import generate_report
from deepsafe_eval.runner import run_inference

# Re-use the existing metrics pipeline to avoid code duplication.
from run_metrics_only import (
    _compute_modality_metrics,
    _load_predictions,
)

_DEFAULT_PREDICTIONS = (
    Path(__file__).parent / "results" / "archive" / "eval_per_file_predictions.json"
)

_MODALITIES = ("images", "audio", "video")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Full DeepSafe evaluation pipeline: "
            "metrics computation + Markdown report generation."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_DEFAULT_PREDICTIONS,
        help="Path to per-file prediction JSON (default: archive copy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("Output directory for results. " "Default: eval/results/"),
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
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Run fresh inference instead of loading from predictions file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "deepsafe_config.json",
        help="Path to deepsafe_config.json (used with --fresh).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset root directory (used with --fresh).",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Discard partial file and restart fresh inference.",
    )
    return parser.parse_args()


def run_pipeline(
    predictions_path: Path,
    output_dir: Path,
    threshold: float = 0.5,
    modality: str | None = None,
    fresh: bool = False,
    config_path: Path | None = None,
    dataset_root: Path | None = None,
    force_restart: bool = False,
) -> tuple[Path, Path]:
    """Execute the full evaluation pipeline.

    Loads predictions, computes all metrics for each modality,
    saves a JSON results file, and generates a Markdown report.

    Args:
        predictions_path: Path to the per-file prediction JSON.
        output_dir: Directory for output files.
        threshold: Classification threshold for binary decisions.
        modality: Optional single modality to evaluate.
        fresh: If True, run fresh inference instead of loading file.
        config_path: Path to deepsafe_config.json (used with fresh).
        dataset_root: Dataset root directory (used with fresh).
        force_restart: Discard partial file and restart inference.

    Returns:
        Tuple of (json_path, markdown_path) for the saved outputs.
    """
    print("=" * 70)
    print("  DeepSafe Full Evaluation Pipeline")
    print(f"  Predictions: {predictions_path}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Threshold:   {threshold}")
    print("=" * 70)

    # Step 1: Get predictions (fresh inference or load from file)
    if fresh:
        print("\n  [1/3] Running fresh inference...")
        raw_predictions = run_inference(
            config_path=config_path,
            dataset_root=dataset_root,
            output_dir=output_dir,
            force_restart=force_restart,
        )
        # Convert list of predictions to modality-grouped dict.
        # build_prediction uses media_type ("image") but metrics
        # pipeline expects metadata keys ("images").
        _MEDIA_TO_MODALITY = {"image": "images", "audio": "audio", "video": "video"}
        data: dict[str, list] = {}
        for pred in raw_predictions:
            media_type = pred.get("modality")
            if media_type is not None:
                mod_key = _MEDIA_TO_MODALITY.get(media_type, media_type)
            else:
                prefix = pred["id"].split("_")[0]
                mod_key = {"img": "images", "aud": "audio", "vid": "video"}.get(
                    prefix, "images"
                )
            data.setdefault(mod_key, []).append(pred)
    else:
        print("\n  [1/3] Loading predictions...")
        data = _load_predictions(predictions_path)

    modalities = [modality] if modality else list(_MODALITIES)
    timestamp = time.strftime("%Y-%m-%d-%H-%M")

    # Step 2: Compute metrics for each modality
    print("\n  [2/3] Computing metrics...")
    results = {
        "timestamp": timestamp,
        "source": str(predictions_path),
        "metadata": {
            "predictions_file": str(predictions_path),
            "threshold": threshold,
            "timestamp": timestamp,
        },
    }

    for mod in modalities:
        predictions = data.get(mod, [])
        if not predictions:
            print(f"    [{mod}] No predictions found -- skipping.")
            continue

        n_real = sum(1 for p in predictions if p["label"] == 0)
        n_fake = sum(1 for p in predictions if p["label"] == 1)
        print(
            f"    [{mod}] {len(predictions)} samples "
            f"({n_real} real / {n_fake} fake)..."
        )

        metrics = _compute_modality_metrics(predictions, threshold)
        results[mod] = metrics

        # Print a brief summary line for each modality.
        ens = metrics.get("ensemble", {})
        if "error" not in ens:
            auc = ens.get("auc_roc", 0)
            eer = ens.get("eer", 0)
            f1 = ens.get("f1", 0)
            print(f"           AUC={auc:.4f}  EER={eer:.4f}  F1={f1:.4f}")

    # Step 3: Save outputs
    print("\n  [3/3] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results
    json_path = output_dir / f"{timestamp}-metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    JSON  -> {json_path}")

    # Markdown report
    md_path = output_dir / f"{timestamp}-report.md"
    generate_report(results, md_path)
    print(f"    Report -> {md_path}")

    print("\n  Done.")
    return json_path, md_path


def main() -> None:
    """CLI entry point for the full evaluation pipeline."""
    args = _parse_args()

    output_dir = args.output_dir if args.output_dir else RESULTS_DIR
    run_pipeline(
        predictions_path=args.predictions,
        output_dir=output_dir,
        threshold=args.threshold,
        modality=args.modality,
        fresh=args.fresh,
        config_path=args.config,
        dataset_root=args.dataset_root,
        force_restart=args.force_restart,
    )


if __name__ == "__main__":
    main()
