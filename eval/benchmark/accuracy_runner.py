#!/usr/bin/env python3
"""Accuracy evaluation runner for GPU benchmark waves.

Sends every file in master_eval_full through each detection model for a
given modality and records per-file predictions with crash-safe
incremental saves.

Usage:
    python accuracy_runner.py --modality image
    python accuracy_runner.py --modality audio --resume
    python accuracy_runner.py --modality video --resume
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow importing the sibling models module.
sys.path.insert(0, str(Path(__file__).parent))
from models import get_models_by_modality  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATASET_ROOT = Path(
    os.getenv("DEEPSAFE_DATASET", str(_PROJECT_ROOT / "dataset" / "master_eval_full"))
)

_RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"

# Modality CLI arg -> dataset subdirectory name.
_MODALITY_SUBDIR = {
    "image": "images",
    "audio": "audio",
    "video": "video",
}

# Accepted file extensions per modality.
_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"},
    "audio": {".mp3", ".wav", ".flac", ".aac", ".ogg"},
    "video": {".mp4", ".avi", ".mov", ".mkv"},
}

# Timeout per modality (seconds).
_TIMEOUTS = {
    "image": 300,
    "audio": 300,
    "video": 600,
}

# How often to flush predictions to disk.
_SAVE_INTERVAL = 25

# How often to print progress.
_PROGRESS_INTERVAL = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_files(modality: str) -> list[dict]:
    """Walk the dataset and collect all files for the given modality.

    The dataset is organised as:
        master_eval_full/{images,audio,video}/{real,fake}/{generator}/file

    Args:
        modality: One of "image", "audio", "video".

    Returns:
        List of dicts with keys: path, label, label_int, generator,
        file_size_bytes.

    Raises:
        SystemExit: If the dataset directory does not exist.
    """
    subdir = _MODALITY_SUBDIR[modality]
    modality_dir = _DATASET_ROOT / subdir
    if not modality_dir.is_dir():
        logger.error("Dataset directory not found: %s", modality_dir)
        sys.exit(1)

    valid_exts = _EXTENSIONS[modality]
    files: list[dict] = []

    for label_name in ("real", "fake"):
        label_dir = modality_dir / label_name
        if not label_dir.is_dir():
            logger.warning("Label directory missing: %s", label_dir)
            continue

        label_int = 0 if label_name == "real" else 1

        for generator_dir in sorted(label_dir.iterdir()):
            if not generator_dir.is_dir():
                continue
            generator = generator_dir.name

            for filepath in sorted(generator_dir.rglob("*")):
                if not filepath.is_file():
                    continue
                if filepath.suffix.lower() not in valid_exts:
                    continue
                files.append({
                    "path": str(filepath),
                    "label": label_name,
                    "label_int": label_int,
                    "generator": generator,
                    "file_size_bytes": filepath.stat().st_size,
                })

    logger.info(
        "Collected %d %s files from %s",
        len(files),
        modality,
        modality_dir,
    )
    return files


# ---------------------------------------------------------------------------
# Model calling
# ---------------------------------------------------------------------------


def call_model(
    url: str,
    payload_key: str,
    b64_data: str,
    timeout: int,
) -> dict:
    """POST base64 payload to a model's /predict endpoint.

    Args:
        url: Full model URL (e.g. http://localhost:5001/predict).
        payload_key: JSON key for the payload (image_data, etc.).
        b64_data: Base64-encoded file content.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: probability (float or None),
        prediction (str or None), latency_ms (float),
        error (str or None).
    """
    payload = {payload_key: b64_data, "threshold": 0.5}
    t0 = time.monotonic()

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        if resp.status_code != 200:
            return {
                "probability": None,
                "prediction": None,
                "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }

        data = resp.json()

        # Extract probability from whichever key the model uses.
        prob = None
        for key in ("probability", "fake_probability", "score"):
            if key in data and isinstance(data[key], (int, float)):
                prob = float(data[key])
                break

        # Extract prediction string.
        prediction = data.get("prediction")
        if prediction is None and prob is not None:
            prediction = "fake" if prob >= 0.5 else "real"

        return {
            "probability": prob,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "error": None,
        }
    except requests.exceptions.Timeout:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        return {
            "probability": None,
            "prediction": None,
            "latency_ms": latency_ms,
            "error": f"Timeout after {timeout}s",
        }
    except requests.exceptions.ConnectionError as exc:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        return {
            "probability": None,
            "prediction": None,
            "latency_ms": latency_ms,
            "error": f"ConnectionError: {str(exc)[:200]}",
        }
    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000, 2)
        return {
            "probability": None,
            "prediction": None,
            "latency_ms": latency_ms,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


# ---------------------------------------------------------------------------
# Incremental save / resume
# ---------------------------------------------------------------------------


def _output_path(modality: str) -> Path:
    """Return the predictions output file path for a modality.

    Args:
        modality: One of "image", "audio", "video".

    Returns:
        Path to the predictions JSON file.
    """
    return _RESULTS_DIR / f"{modality}_predictions.json"


def save_predictions(
    modality: str,
    models_list: list[str],
    predictions: list[dict],
    total_files: int,
) -> None:
    """Write the full predictions list to disk.

    Args:
        modality: The modality being evaluated.
        models_list: Sorted list of model names used.
        predictions: List of per-file prediction dicts.
        total_files: Total number of files in the dataset.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "models": models_list,
        "modality": modality,
        "total_files": total_files,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "predictions": predictions,
    }
    out_path = _output_path(modality)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def load_completed_paths(modality: str) -> set[str]:
    """Load previously completed file paths for resume.

    Args:
        modality: The modality to resume.

    Returns:
        Set of file path strings already processed.
    """
    out_path = _output_path(modality)
    if not out_path.exists():
        return set()

    try:
        with open(out_path) as f:
            data = json.load(f)
        completed = {p["path"] for p in data.get("predictions", [])}
        logger.info(
            "Resume: loaded %d completed predictions from %s",
            len(completed),
            out_path,
        )
        return completed
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        logger.warning(
            "Could not load existing predictions for resume: %s", exc
        )
        return set()


def load_existing_predictions(modality: str) -> list[dict]:
    """Load the existing predictions list for resume.

    Args:
        modality: The modality to resume.

    Returns:
        List of prediction dicts from the previous run, or empty list.
    """
    out_path = _output_path(modality)
    if not out_path.exists():
        return []

    try:
        with open(out_path) as f:
            data = json.load(f)
        return data.get("predictions", [])
    except (json.JSONDecodeError, KeyError, OSError):
        return []


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_accuracy(modality: str, resume: bool = False) -> None:
    """Run accuracy evaluation for a modality.

    Processes every file through every model for the given modality,
    recording per-file predictions with crash-safe incremental saves.

    Args:
        modality: One of "image", "audio", "video".
        resume: If True, skip files already in the predictions file.
    """
    # Discover models.
    models = get_models_by_modality(modality)
    if not models:
        logger.error("No models found for modality: %s", modality)
        sys.exit(1)

    models_list = sorted(m.name for m in models)
    timeout = _TIMEOUTS[modality]

    logger.info(
        "Modality: %s | Models: %s | Timeout: %ds",
        modality,
        ", ".join(models_list),
        timeout,
    )

    # Collect files.
    all_files = collect_files(modality)
    total_files = len(all_files)
    if total_files == 0:
        logger.error("No files found for modality: %s", modality)
        sys.exit(1)

    # Resume support.
    if resume:
        completed_paths = load_completed_paths(modality)
        predictions = load_existing_predictions(modality)
        pending_files = [
            f for f in all_files if f["path"] not in completed_paths
        ]
        logger.info(
            "Resume: %d already done, %d remaining",
            len(completed_paths),
            len(pending_files),
        )
    else:
        predictions = []
        pending_files = all_files

    if not pending_files:
        logger.info("All files already processed. Nothing to do.")
        return

    # Pre-flight health check: skip models that are not responding.
    healthy_models = []
    for model in models:
        try:
            r = requests.get(
                f"http://localhost:{model.port}/health", timeout=10,
            )
            if r.status_code == 200:
                healthy_models.append(model)
                logger.info("  %s (:%d): HEALTHY", model.name, model.port)
            else:
                logger.warning(
                    "  %s (:%d): HTTP %d -- SKIPPING", model.name, model.port, r.status_code,
                )
        except Exception:
            logger.warning(
                "  %s (:%d): NOT RESPONDING -- SKIPPING", model.name, model.port,
            )
    if not healthy_models:
        logger.error("No healthy models found. Aborting.")
        sys.exit(1)
    models = healthy_models
    logger.info("Running with %d/%d healthy models", len(models), len(models))

    start_time = time.monotonic()
    start_idx = total_files - len(pending_files)

    for i, file_info in enumerate(pending_files):
        global_idx = start_idx + i + 1
        file_path = Path(file_info["path"])

        # Progress logging.
        if global_idx % _PROGRESS_INTERVAL == 0 or global_idx == 1:
            print(
                f"[{global_idx}/{total_files}] Processing "
                f"{file_path.name}...",
                flush=True,
            )

        # Read and encode the file.
        try:
            raw_bytes = file_path.read_bytes()
            b64_data = base64.b64encode(raw_bytes).decode()
        except (OSError, IOError) as exc:
            logger.warning(
                "[%d/%d] Skipping %s — file read error: %s",
                global_idx,
                total_files,
                file_path.name,
                exc,
            )
            continue

        # Call each model sequentially.
        model_results = {}
        for model in models:
            url = f"http://localhost:{model.port}/predict"
            result = call_model(
                url=url,
                payload_key=model.payload_key,
                b64_data=b64_data,
                timeout=timeout,
            )
            model_results[model.name] = result

        # Build prediction record.
        prediction = {
            "path": str(file_path),
            "label": file_info["label"],
            "label_int": file_info["label_int"],
            "generator": file_info["generator"],
            "model_results": model_results,
            "file_size_bytes": file_info["file_size_bytes"],
        }
        predictions.append(prediction)

        # Crash-safe incremental save.
        if len(predictions) % _SAVE_INTERVAL == 0:
            save_predictions(modality, models_list, predictions, total_files)
            logger.info(
                "Saved checkpoint: %d/%d predictions",
                len(predictions),
                total_files,
            )

    # Final save.
    save_predictions(modality, models_list, predictions, total_files)

    elapsed = time.monotonic() - start_time
    n_processed = len(pending_files)
    rate = n_processed / elapsed if elapsed > 0 else 0.0

    logger.info(
        "Done: %d files processed in %.1fs (%.1f files/s). "
        "Total predictions: %d/%d.",
        n_processed,
        elapsed,
        rate,
        len(predictions),
        total_files,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the accuracy evaluation."""
    parser = argparse.ArgumentParser(
        description="Run accuracy evaluation on master_eval_full dataset.",
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=["image", "audio", "video"],
        help="Modality to evaluate.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing predictions file, skipping completed.",
    )
    parser.add_argument(
        "--skip-models",
        type=str,
        default="",
        help="Comma-separated model names to skip (e.g., 'fakestormer,dfd_fcg').",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Override default timeout (seconds). 0 = use defaults.",
    )
    args = parser.parse_args()

    # Filter out skipped models by temporarily modifying the models list
    skip_set = set(args.skip_models.split(",")) if args.skip_models else set()
    if skip_set:
        from models import ALL_MODELS
        original = [m for m in ALL_MODELS if m.modality == args.modality]
        active = [m.name for m in original if m.name not in skip_set]
        print(f"Skipping models: {skip_set}")
        print(f"Active models: {active}")

    run_accuracy(modality=args.modality, resume=args.resume)


if __name__ == "__main__":
    main()
