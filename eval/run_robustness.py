#!/usr/bin/env python3
"""Robustness testing: measure detection survival under degradation.

Selects N fake samples from the dataset, applies various degradation
attacks (JPEG recompression, social-media simulation, Gaussian noise,
blur for images; MP3 re-encoding for audio; CRF re-encoding for video),
then queries model services directly to measure detection rate at each
degradation level.

Samples are processed in batches (default 5) with explicit garbage
collection between batches to avoid OOM on memory-constrained systems
(e.g., Mac with limited RAM).

Usage:
    uv run python -m eval.run_robustness --modality image --n-samples 50
    uv run python -m eval.run_robustness --modality audio --n-samples 30
    uv run python -m eval.run_robustness --modality video --n-samples 20
    uv run python -m eval.run_robustness --modality image --n-samples 50 --batch-size 3
"""

import argparse
import base64
import gc
import json
import random
import sys
import time
from pathlib import Path

import requests
from deepsafe_eval.config import (
    AUDIO_MODELS,
    DATASET_ROOT,
    IMAGE_MODELS,
    MODEL_TIMEOUT,
    RESULTS_DIR,
    VIDEO_MODELS,
)
from deepsafe_eval.robustness import (
    add_gaussian_noise,
    gaussian_blur,
    jpeg_recompress,
    mp3_reencode,
    social_media_simulate,
    video_reencode,
)

# --- Degradation Configurations ---

IMAGE_ATTACKS: list[dict] = [
    {"name": "original", "fn": None, "params": {}},
    {"name": "jpeg_q80", "fn": jpeg_recompress, "params": {"quality": 80}},
    {"name": "jpeg_q60", "fn": jpeg_recompress, "params": {"quality": 60}},
    {"name": "jpeg_q40", "fn": jpeg_recompress, "params": {"quality": 40}},
    {"name": "jpeg_q20", "fn": jpeg_recompress, "params": {"quality": 20}},
    {"name": "noise_sigma5", "fn": add_gaussian_noise, "params": {"sigma": 5.0}},
    {"name": "noise_sigma10", "fn": add_gaussian_noise, "params": {"sigma": 10.0}},
    {"name": "noise_sigma20", "fn": add_gaussian_noise, "params": {"sigma": 20.0}},
    {"name": "blur_k3", "fn": gaussian_blur, "params": {"kernel_size": 3}},
    {"name": "blur_k5", "fn": gaussian_blur, "params": {"kernel_size": 5}},
    {"name": "blur_k9", "fn": gaussian_blur, "params": {"kernel_size": 9}},
    {
        "name": "social_instagram",
        "fn": social_media_simulate,
        "params": {"platform": "instagram"},
    },
    {
        "name": "social_twitter",
        "fn": social_media_simulate,
        "params": {"platform": "twitter"},
    },
    {
        "name": "social_whatsapp",
        "fn": social_media_simulate,
        "params": {"platform": "whatsapp"},
    },
]

AUDIO_ATTACKS: list[dict] = [
    {"name": "original", "fn": None, "params": {}},
    {"name": "mp3_192k", "fn": mp3_reencode, "params": {"bitrate": "192k"}},
    {"name": "mp3_128k", "fn": mp3_reencode, "params": {"bitrate": "128k"}},
    {"name": "mp3_64k", "fn": mp3_reencode, "params": {"bitrate": "64k"}},
    {"name": "mp3_32k", "fn": mp3_reencode, "params": {"bitrate": "32k"}},
]

VIDEO_ATTACKS: list[dict] = [
    {"name": "original", "fn": None, "params": {}},
    {"name": "crf_18", "fn": video_reencode, "params": {"crf": 18}},
    {"name": "crf_23", "fn": video_reencode, "params": {"crf": 23}},
    {"name": "crf_28", "fn": video_reencode, "params": {"crf": 28}},
    {"name": "crf_35", "fn": video_reencode, "params": {"crf": 35}},
    {"name": "crf_40", "fn": video_reencode, "params": {"crf": 40}},
]

_MODALITY_DIR_MAP = {"image": "images", "audio": "audio", "video": "video"}


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Robustness testing under degradation attacks."
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["image", "audio", "video"],
        help="Modality to test.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of fake samples to test (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for results JSON (default: auto-timestamped).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to test (default: all for modality).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help=(
            "Number of samples to process per batch before running "
            "garbage collection (default: 5). Lower values reduce "
            "peak memory on constrained systems."
        ),
    )
    return parser.parse_args()


def _collect_fake_files(
    modality: str,
    n_samples: int,
    seed: int,
) -> list[tuple[Path, str]]:
    """Collect fake sample files from the dataset.

    Args:
        modality: One of 'image', 'audio', 'video'.
        n_samples: Number of samples to select.
        seed: Random seed.

    Returns:
        List of (file_path, generator_name) tuples.
    """
    fs_dir = _MODALITY_DIR_MAP[modality]
    fake_dir = DATASET_ROOT / fs_dir / "fake"
    if not fake_dir.exists():
        print(f"ERROR: Fake directory not found: {fake_dir}", file=sys.stderr)
        return []

    items: list[tuple[Path, str]] = []
    for gen_dir in sorted(fake_dir.iterdir()):
        if not gen_dir.is_dir():
            continue
        gen_name = gen_dir.name
        for f in sorted(gen_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                items.append((f, gen_name))

    if not items:
        return []

    rng = random.Random(seed)
    if len(items) > n_samples:
        items = rng.sample(items, n_samples)
    return items


def _b64_bytes(data: bytes) -> str:
    """Encode raw bytes as base64 string."""
    return base64.b64encode(data).decode()


def _b64_file(path: Path) -> str:
    """Read a file and return base64-encoded content."""
    return base64.b64encode(path.read_bytes()).decode()


def _check_model_health(name: str, url: str) -> bool:
    """Check if a model service is online.

    Args:
        name: Model display name.
        url: Prediction endpoint URL.

    Returns:
        True if the health endpoint responds 200.
    """
    health_url = url.replace("/predict", "/health")
    try:
        resp = requests.get(health_url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _call_model_with_bytes(
    url: str,
    data_key: str,
    payload_b64: str,
) -> tuple[float | None, str | None]:
    """Send base64 payload to a model and extract the probability.

    Args:
        url: Model prediction endpoint.
        data_key: JSON key for the base64 data.
        payload_b64: Base64-encoded media bytes.

    Returns:
        Tuple of (probability, error_message).
    """
    try:
        resp = requests.post(
            url,
            json={data_key: payload_b64},
            timeout=MODEL_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        for key in ("probability", "fake_probability", "score"):
            if key in data and isinstance(data[key], (int, float)):
                return float(data[key]), None
        return None, f"unexpected keys: {list(data.keys())}"
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, str(e)[:120]


def _get_data_key(modality: str) -> str:
    """Return the JSON payload key for the given modality."""
    return {
        "image": "image_data",
        "audio": "audio_data",
        "video": "video_data",
    }[modality]


def _run_robustness_test(
    modality: str,
    samples: list[tuple[Path, str]],
    models: dict[str, str],
    attacks: list[dict],
    threshold: float,
    batch_size: int = 5,
) -> dict:
    """Run the full robustness test.

    For each sample and each attack, applies the degradation and queries
    every live model.  Collects detection rates per attack per model.
    Samples are processed in batches with explicit gc.collect() calls
    between batches to keep peak memory bounded on constrained systems.

    Args:
        modality: Modality name.
        samples: List of (path, generator) tuples.
        models: Dict of model_name -> endpoint URL.
        attacks: List of attack configuration dicts.
        threshold: Classification threshold.
        batch_size: Number of samples per batch before GC (default 5).

    Returns:
        Full results dict with per-model, per-attack detection rates.
    """
    data_key = _get_data_key(modality)

    # Check which models are online.
    live_models: dict[str, str] = {}
    for name, url in models.items():
        if _check_model_health(name, url):
            live_models[name] = url
            print(f"    [{name}] ONLINE", flush=True)
        else:
            print(f"    [{name}] OFFLINE -- skipping", flush=True)

    if not live_models:
        print("  No models available. Aborting.", flush=True)
        return {}

    # Results: model -> attack -> {detected, total, errors, detection_rate}
    results: dict[str, dict[str, dict]] = {m: {} for m in live_models}

    total_ops = len(samples) * len(attacks) * len(live_models)
    completed = 0

    for attack in attacks:
        attack_name = attack["name"]
        attack_fn = attack["fn"]
        attack_params = attack["params"]

        print(f"\n  Attack: {attack_name}", flush=True)

        for model_name, url in live_models.items():
            detected = 0
            errors = 0
            total = 0

            for idx, (file_path, generator) in enumerate(samples):
                total += 1
                completed += 1

                try:
                    if attack_fn is None:
                        # Original -- read file directly.
                        payload_b64 = _b64_file(file_path)
                    else:
                        degraded_bytes = attack_fn(file_path, **attack_params)
                        payload_b64 = _b64_bytes(degraded_bytes)
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(
                            f"    [{model_name}] Degradation error on "
                            f"{file_path.name}: {e}",
                            flush=True,
                        )
                    continue

                prob, err = _call_model_with_bytes(url, data_key, payload_b64)
                if prob is not None:
                    if prob >= threshold:
                        detected += 1
                else:
                    errors += 1

                if completed % 50 == 0:
                    print(
                        f"    Progress: {completed}/{total_ops}",
                        flush=True,
                    )

                # Free memory after each batch to avoid OOM.
                if (idx + 1) % batch_size == 0:
                    del payload_b64
                    gc.collect()

            det_rate = detected / (total - errors) if (total - errors) > 0 else 0.0
            results[model_name][attack_name] = {
                "detected": detected,
                "total": total,
                "errors": errors,
                "detection_rate": round(det_rate, 4),
            }
            print(
                f"    [{model_name}] {attack_name}: "
                f"{detected}/{total - errors} detected "
                f"({det_rate:.1%}) "
                f"[{errors} errors]",
                flush=True,
            )

    return results


def _print_summary(results: dict, attacks: list[dict]) -> None:
    """Print a summary table of detection rates per attack per model.

    Args:
        results: Per-model, per-attack results dict.
        attacks: List of attack configurations (for ordering).
    """
    if not results:
        return

    attack_names = [a["name"] for a in attacks]
    model_names = sorted(results.keys())

    print(f"\n{'='*80}")
    print("  ROBUSTNESS SUMMARY -- Detection Rate (%) per Attack")
    print(f"{'='*80}")

    # Header
    header = f"  {'Attack':<22}"
    for m in model_names:
        header += f" {m:>12}"
    print(header)
    print(f"  {'-'*(22 + 13 * len(model_names))}")

    for attack_name in attack_names:
        row = f"  {attack_name:<22}"
        for m in model_names:
            entry = results[m].get(attack_name, {})
            rate = entry.get("detection_rate")
            if rate is not None:
                row += f" {rate*100:>11.1f}%"
            else:
                row += f" {'---':>12}"
        print(row)

    # Delta from original
    print(f"\n  Delta from original (percentage points):")
    print(f"  {'-'*(22 + 13 * len(model_names))}")
    for attack_name in attack_names:
        if attack_name == "original":
            continue
        row = f"  {attack_name:<22}"
        for m in model_names:
            orig = results[m].get("original", {}).get("detection_rate")
            curr = results[m].get(attack_name, {}).get("detection_rate")
            if orig is not None and curr is not None:
                delta = (curr - orig) * 100
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:>10.1f}pp"
            else:
                row += f" {'---':>12}"
        print(row)


def main() -> None:
    """Run the robustness testing pipeline."""
    args = _parse_args()

    modality = args.modality
    n_samples = args.n_samples
    threshold = args.threshold

    batch_size = args.batch_size

    print("=" * 80)
    print("  DeepSafe Robustness Testing")
    print(f"  Modality:    {modality}")
    print(f"  Samples:     {n_samples}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Threshold:   {threshold}")
    print(f"  Seed:        {args.seed}")
    print("=" * 80)

    # Select models and attacks based on modality.
    if modality == "image":
        all_models = dict(IMAGE_MODELS)
        attacks = IMAGE_ATTACKS
    elif modality == "audio":
        all_models = dict(AUDIO_MODELS)
        attacks = AUDIO_ATTACKS
    elif modality == "video":
        all_models = dict(VIDEO_MODELS)
        attacks = VIDEO_ATTACKS
    else:
        print(f"ERROR: Unknown modality: {modality}", file=sys.stderr)
        sys.exit(1)

    # Apply --models filter.
    if args.models:
        model_filter = set(args.models.split(","))
        all_models = {k: v for k, v in all_models.items() if k in model_filter}

    if not all_models:
        print("ERROR: No models selected.", file=sys.stderr)
        sys.exit(1)

    # Collect fake samples.
    print(f"\n  Collecting {n_samples} fake samples...", flush=True)
    samples = _collect_fake_files(modality, n_samples, args.seed)
    if not samples:
        print("ERROR: No fake samples found.", file=sys.stderr)
        sys.exit(1)

    # Show generator distribution.
    gen_counts: dict[str, int] = {}
    for _, gen in samples:
        gen_counts[gen] = gen_counts.get(gen, 0) + 1
    print(f"  Selected {len(samples)} samples from " f"{len(gen_counts)} generators:")
    for gen in sorted(gen_counts.keys()):
        print(f"    {gen}: {gen_counts[gen]}")

    # Check models.
    print(f"\n  Checking {len(all_models)} model(s)...", flush=True)

    # Run the test.
    t0 = time.time()
    results = _run_robustness_test(
        modality, samples, all_models, attacks, threshold, batch_size
    )
    elapsed = time.time() - t0

    if not results:
        print("  No results collected.", file=sys.stderr)
        sys.exit(1)

    # Print summary.
    _print_summary(results, attacks)
    print(f"\n  Total time: {elapsed:.0f}s")

    # Build report.
    report = {
        "metadata": {
            "modality": modality,
            "n_samples": len(samples),
            "threshold": threshold,
            "seed": args.seed,
            "elapsed_s": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_root": str(DATASET_ROOT),
        },
        "generator_distribution": gen_counts,
        "attacks": [a["name"] for a in attacks],
        "results": results,
    }

    # Save JSON.
    if args.output:
        out_path = args.output
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d-%H-%M")
        out_path = RESULTS_DIR / f"{stamp}-robustness-{modality}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Results saved -> {out_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
