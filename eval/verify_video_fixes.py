#!/usr/bin/env python3
"""Verify fixes for 4 video models: FakeSTormer, RECCE, PwTF-DVD, UnivFD-Video.

Sends a small set of known-label video files to the monolith server and checks
that each model now produces sensible outputs. Run this AFTER applying the
fixes from commit fd33f59 and restarting the monolith server.

Prerequisites:
    - Monolith server running: python -m monolith.server
    - At least these 4 models loaded: fakestormer, recce, pwtf_dvd, univfd_video
    - Video test files in the dataset (see --dataset-dir)

Usage:
    # Quick smoke test (5 real + 5 fake videos, ~2 min)
    python eval/verify_video_fixes.py

    # Full verification (50 real + 50 fake, ~20 min)
    python eval/verify_video_fixes.py --num-samples 50

    # Custom server URL
    python eval/verify_video_fixes.py --server http://10.0.0.5:8000

    # Custom dataset path
    python eval/verify_video_fixes.py --dataset-dir /data/master_eval_full/video

What this script checks per model:
    1. Model is loaded and responding (health check)
    2. Inference succeeds without crashes (probability != null)
    3. Fake videos score HIGHER than real videos on average (correct direction)
    4. AUC > 0.5 (above chance, not anti-correlated)

Expected results after fixes:
    - FakeSTormer: was crashing 99.9%, should now return valid probabilities
    - RECCE:       was AUC=0.332 (inverted), should now be ~0.668
    - PwTF-DVD:    was AUC=0.417 (inverted), should now be ~0.583
    - UnivFD-Video: was AUC=0.469 (bad transform), should improve
"""

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger("verify_video_fixes")

# The 4 models we're verifying
TARGET_MODELS = ["fakestormer", "recce", "pwtf_dvd", "univfd_video"]

# Pre-fix AUCs from the medium eval (15.5K samples) — our baseline.
# After fixes, every model should beat its pre-fix AUC.
PREFIX_AUCS = {
    "fakestormer": 0.500,   # crashed on 99.9% of videos
    "recce": 0.332,         # class index was inverted
    "pwtf_dvd": 0.417,      # sigmoid was inverted
    "univfd_video": 0.469,  # Resize(224) destroyed signal
}


def check_server_health(server_url: str) -> dict:
    """Verify the monolith server is up and our 4 models are loaded.

    Returns:
        Dict mapping model name -> bool (loaded).

    Raises:
        ConnectionError: If the server is not reachable.
    """
    try:
        resp = requests.get(f"{server_url}/health", timeout=10)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot reach monolith server at {server_url}. "
            "Start it with: python -m monolith.server"
        )

    health = resp.json()
    models_status = {}
    loaded_models = health.get("models", {})

    for name in TARGET_MODELS:
        info = loaded_models.get(name, {})
        is_loaded = info.get("loaded", False)
        models_status[name] = is_loaded
        status = "OK" if is_loaded else "NOT LOADED"
        logger.info("  %-15s %s", name, status)

    return models_status


def find_video_files(
    dataset_dir: Path,
    num_samples: int,
) -> tuple[list[Path], list[Path]]:
    """Find real and fake video files in the dataset directory.

    Expects directory structure:
        dataset_dir/real/   (or dataset_dir/0_real/)
        dataset_dir/fake/   (or dataset_dir/1_fake/)

    Falls back to searching recursively if standard structure
    is not found.

    Returns:
        (real_files, fake_files) — each up to num_samples.
    """
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # Try standard directory structures
    real_dirs = ["real", "0_real", "Real"]
    fake_dirs = ["fake", "1_fake", "Fake"]

    real_root = None
    fake_root = None

    for d in real_dirs:
        p = dataset_dir / d
        if p.exists():
            real_root = p
            break

    for d in fake_dirs:
        p = dataset_dir / d
        if p.exists():
            fake_root = p
            break

    def collect_videos(root: Path, limit: int) -> list[Path]:
        files = []
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in video_exts and f.is_file():
                files.append(f)
                if len(files) >= limit:
                    break
        return files

    if real_root and fake_root:
        reals = collect_videos(real_root, num_samples)
        fakes = collect_videos(fake_root, num_samples)
    else:
        # Fallback: look for any videos and warn
        logger.warning(
            "No real/fake subdirectories found in %s. "
            "Searching recursively for any video files.",
            dataset_dir,
        )
        all_vids = collect_videos(dataset_dir, num_samples * 2)
        mid = len(all_vids) // 2
        reals = all_vids[:mid]
        fakes = all_vids[mid:]
        logger.warning(
            "WARNING: Cannot determine labels. Treating first %d as "
            "real, last %d as fake. Results will be unreliable.",
            len(reals), len(fakes),
        )

    return reals, fakes


def predict_single_model(
    server_url: str,
    model_name: str,
    video_path: Path,
    timeout: int = 120,
) -> dict:
    """Send a video file to a single model endpoint and return the result.

    Args:
        server_url: Base URL of the monolith server.
        model_name: Model name (e.g., "fakestormer").
        video_path: Path to the video file.
        timeout: Request timeout in seconds (video models can be slow).

    Returns:
        Dict with "probability", "prediction", "latency_ms", and
        optionally "error" if inference failed.
    """
    video_bytes = video_path.read_bytes()
    b64_data = base64.b64encode(video_bytes).decode()

    payload = {"video_data": b64_data}
    url = f"{server_url}/models/{model_name}/predict"

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"probability": None, "error": "timeout"}
    except Exception as e:
        return {"probability": None, "error": str(e)}


def run_verification(
    server_url: str,
    real_files: list[Path],
    fake_files: list[Path],
    models: list[str],
) -> dict:
    """Run all models on all files and collect per-model results.

    Returns:
        Dict mapping model_name -> {
            "real_probs": [...],
            "fake_probs": [...],
            "errors": int,
            "total": int,
        }
    """
    results = {
        m: {"real_probs": [], "fake_probs": [], "errors": 0, "total": 0}
        for m in models
    }

    all_files = [(f, "real") for f in real_files] + [
        (f, "fake") for f in fake_files
    ]
    total = len(all_files)

    for i, (vfile, label) in enumerate(all_files):
        logger.info(
            "[%d/%d] %s: %s", i + 1, total, label, vfile.name,
        )

        for model in models:
            result = predict_single_model(server_url, model, vfile)
            prob = result.get("probability")
            results[model]["total"] += 1

            if prob is None:
                results[model]["errors"] += 1
                logger.warning(
                    "  %-15s ERROR: %s",
                    model, result.get("error", "null probability"),
                )
            else:
                if label == "real":
                    results[model]["real_probs"].append(prob)
                else:
                    results[model]["fake_probs"].append(prob)

                lat = result.get("latency_ms", "?")
                logger.info(
                    "  %-15s prob=%.4f  lat=%sms", model, prob, lat,
                )

    return results


def compute_auc(real_probs: list, fake_probs: list) -> float:
    """Compute AUC from separate real/fake probability lists.

    Uses the simple Wilcoxon-Mann-Whitney statistic: the fraction
    of (fake, real) pairs where the fake score > real score.
    Equivalent to sklearn's roc_auc_score but without the dependency.
    """
    if not real_probs or not fake_probs:
        return 0.5

    concordant = 0
    tied = 0
    total = len(real_probs) * len(fake_probs)

    for fp in fake_probs:
        for rp in real_probs:
            if fp > rp:
                concordant += 1
            elif fp == rp:
                tied += 1

    return (concordant + 0.5 * tied) / total


def print_report(results: dict) -> bool:
    """Print a summary report and return True if all models pass.

    A model passes if:
        1. Error rate < 20%
        2. Mean fake score > mean real score (correct direction)
        3. AUC > 0.50 (above chance)
    """
    print("\n" + "=" * 72)
    print("VERIFICATION REPORT — Video Model Fixes")
    print("=" * 72)

    all_pass = True

    for model in TARGET_MODELS:
        r = results.get(model)
        if r is None:
            print(f"\n{model}: SKIPPED (not loaded)")
            all_pass = False
            continue

        n_real = len(r["real_probs"])
        n_fake = len(r["fake_probs"])
        errors = r["errors"]
        total = r["total"]
        error_rate = errors / total if total > 0 else 1.0

        print(f"\n{'─' * 72}")
        print(f"  {model.upper()}")
        print(f"{'─' * 72}")
        print(f"  Samples:    {total} total ({n_real} real, {n_fake} fake, {errors} errors)")
        print(f"  Error rate: {error_rate:.1%}")

        if n_real == 0 or n_fake == 0:
            print("  FAIL: not enough valid predictions to evaluate")
            all_pass = False
            continue

        mean_real = np.mean(r["real_probs"])
        mean_fake = np.mean(r["fake_probs"])
        auc = compute_auc(r["real_probs"], r["fake_probs"])
        prefix_auc = PREFIX_AUCS.get(model, 0.5)
        direction_ok = mean_fake > mean_real
        auc_ok = auc > 0.50
        error_ok = error_rate < 0.20

        print(f"  Mean real:  {mean_real:.4f}")
        print(f"  Mean fake:  {mean_fake:.4f}")
        print(f"  Direction:  {'CORRECT (fake > real)' if direction_ok else 'WRONG (real > fake)'}")
        print(f"  AUC:        {auc:.4f}  (pre-fix: {prefix_auc:.3f}, delta: {auc - prefix_auc:+.3f})")

        # Verdict
        checks = {
            "errors < 20%": error_ok,
            "direction correct": direction_ok,
            "AUC > 0.50": auc_ok,
        }
        passed = all(checks.values())
        for check_name, ok in checks.items():
            print(f"    {'PASS' if ok else 'FAIL'}: {check_name}")

        print(f"  VERDICT:    {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    print(f"\n{'=' * 72}")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"{'=' * 72}\n")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Verify fixes for 4 video deepfake detection models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Monolith server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing video/real/ and video/fake/ subdirs. "
            "Defaults to searching common locations."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of real + fake videos to test (default: 5 each)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=TARGET_MODELS,
        help="Models to verify (default: all 4 fixed models)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    logger.info("Verifying video model fixes against %s", args.server)

    # Step 1: Health check
    logger.info("\n--- Health Check ---")
    try:
        status = check_server_health(args.server)
    except ConnectionError as e:
        logger.error(str(e))
        sys.exit(1)

    missing = [m for m in args.models if not status.get(m, False)]
    if missing:
        logger.error(
            "Models not loaded: %s. "
            "Restart server with: DEEPSAFE_MODELS=%s python -m monolith.server",
            ", ".join(missing),
            ",".join(args.models),
        )
        sys.exit(1)

    # Step 2: Find test videos
    logger.info("\n--- Finding Test Videos ---")
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        # Search common locations
        candidates = [
            Path("/deepsafe/dataset/master_eval/video"),
            Path("/data/master_eval_full/master_eval_full/master_eval/video"),
            Path("dataset/master_eval/video"),
        ]
        for c in candidates:
            if c.exists():
                dataset_dir = c
                break

    if dataset_dir is None or not dataset_dir.exists():
        logger.error(
            "No video dataset found. Pass --dataset-dir with a directory "
            "containing real/ and fake/ subdirectories of video files."
        )
        sys.exit(1)

    real_files, fake_files = find_video_files(dataset_dir, args.num_samples)
    logger.info(
        "Found %d real, %d fake videos in %s",
        len(real_files), len(fake_files), dataset_dir,
    )

    if len(real_files) < 2 or len(fake_files) < 2:
        logger.error("Need at least 2 real and 2 fake videos for AUC.")
        sys.exit(1)

    # Step 3: Run predictions
    logger.info("\n--- Running Predictions ---")
    start = time.perf_counter()
    results = run_verification(
        args.server, real_files, fake_files, args.models,
    )
    elapsed = time.perf_counter() - start
    logger.info("Predictions complete in %.1f seconds", elapsed)

    # Step 4: Report
    all_pass = print_report(results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
