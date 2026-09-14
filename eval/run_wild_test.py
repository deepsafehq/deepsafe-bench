#!/usr/bin/env python3
"""Wild test evaluation: benchmark against in-the-wild datasets.

Tests the monolith server against datasets from modern commercial
AI generators not well-represented in the academic eval set:
  - Defactify: SD2.1, SDXL, SD3, DALL-E 3, Midjourney 6 (images)
  - DeepAction v1: RunwayML, Veo, CogVideoX, VideoPoet, etc. (video)
  - garystafford: ElevenLabs voice clones (audio)

Usage:
    # Ensure monolith server is running
    python eval/run_wild_test.py --dataset defactify --max-samples 500
    python eval/run_wild_test.py --dataset deepaction
    python eval/run_wild_test.py --dataset garystafford
    python eval/run_wild_test.py --dataset all --max-samples 200
"""

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

WILD_TEST_DIR = Path(
    os.getenv(
        "DEEPSAFE_WILD_TEST_DIR",
        "/Volumes/16TB_Sid/deepsafe/dataset/wild_test",
    )
)
SERVER_URL = os.getenv("DEEPSAFE_SERVER_URL", "http://localhost:8000")
DETECT_URL = f"{SERVER_URL}/v1/detect"

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp4": "video/mp4", ".avi": "video/avi", ".mov": "video/quicktime",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".m4a": "audio/x-m4a",
}

# Defactify Label_B -> generator name
DEFACTIFY_GENERATORS = {
    0: "real",
    1: "sd_2.1",
    2: "sdxl",
    3: "sd_3",
    4: "dalle_3",
    5: "midjourney_6",
}


# ── Dataset Loaders ─────────────────────────────────────────────────


def load_defactify(max_samples=None):
    """Load Defactify parquet dataset.

    Returns list of dicts with keys:
        image_bytes, label (0=real, 1=fake), generator, modality
    """
    data_dir = WILD_TEST_DIR / "defactify" / "data"
    if not data_dir.exists():
        print(f"Defactify not found at {data_dir}")
        return []

    import pyarrow.parquet as pq
    from PIL import Image

    # Use test split only
    files = sorted(
        f for f in data_dir.iterdir()
        if f.name.startswith("test-") and f.suffix == ".parquet"
    )

    samples = []
    for pf in files:
        table = pq.read_table(pf)
        for i in range(len(table)):
            label_a = table.column("Label_A")[i].as_py()
            label_b = table.column("Label_B")[i].as_py()
            img_data = table.column("Image")[i].as_py()

            # img_data is a dict with 'bytes' and 'path' keys
            if isinstance(img_data, dict):
                img_bytes = img_data.get("bytes", b"")
            elif isinstance(img_data, bytes):
                img_bytes = img_data
            else:
                continue

            if not img_bytes:
                continue

            generator = DEFACTIFY_GENERATORS.get(label_b, f"unknown_{label_b}")
            samples.append({
                "raw_bytes": img_bytes,
                "label": label_a,
                "generator": generator,
                "modality": "image",
                "dataset": "defactify",
                "mime": "image/jpeg",
            })

        if max_samples and len(samples) >= max_samples:
            break

    if max_samples:
        # Stratified subsample: keep proportions per generator
        by_gen = defaultdict(list)
        for s in samples:
            by_gen[s["generator"]].append(s)

        n_gens = len(by_gen)
        per_gen = max(1, max_samples // n_gens)
        subsampled = []
        for gen, gen_samples in by_gen.items():
            rng = np.random.RandomState(42)
            indices = rng.choice(
                len(gen_samples), min(per_gen, len(gen_samples)), replace=False
            )
            subsampled.extend(gen_samples[i] for i in indices)
        samples = subsampled[:max_samples]

    return samples


def load_deepaction(max_samples=None):
    """Load DeepAction v1 video dataset.

    Returns list of dicts with keys:
        file_path, label, generator, modality
    """
    base_dir = WILD_TEST_DIR / "deepaction_v1"
    if not base_dir.exists():
        print(f"DeepAction not found at {base_dir}")
        return []

    real_gen = "real"
    samples = []

    for gen_dir in sorted(base_dir.iterdir()):
        if not gen_dir.is_dir() or gen_dir.name.startswith("."):
            continue

        generator = gen_dir.name.lower()
        label = 0 if generator == "pexels" else 1
        if label == 0:
            generator = real_gen

        videos = sorted(gen_dir.rglob("*.mp4"))
        for vid in videos:
            samples.append({
                "file_path": str(vid),
                "label": label,
                "generator": generator if label == 1 else real_gen,
                "modality": "video",
                "dataset": "deepaction",
                "mime": "video/mp4",
            })

    if max_samples and len(samples) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    return samples


def load_garystafford(max_samples=None):
    """Load garystafford ElevenLabs audio dataset.

    Returns list of dicts with keys:
        file_path, label, generator, modality
    """
    base_dir = WILD_TEST_DIR / "garystafford_audio"
    if not base_dir.exists():
        print(f"garystafford not found at {base_dir}")
        return []

    samples = []

    # Fake (ElevenLabs)
    fake_dir = base_dir / "fake"
    if fake_dir.exists():
        for f in sorted(fake_dir.rglob("*.flac")):
            samples.append({
                "file_path": str(f),
                "label": 1,
                "generator": "elevenlabs",
                "modality": "audio",
                "dataset": "garystafford",
                "mime": "audio/flac",
            })

    # Real (YouTube)
    real_dir = base_dir / "real"
    if real_dir.exists():
        for f in sorted(real_dir.rglob("*.flac")):
            samples.append({
                "file_path": str(f),
                "label": 0,
                "generator": "real",
                "modality": "audio",
                "dataset": "garystafford",
                "mime": "audio/flac",
            })

    if max_samples and len(samples) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    return samples


# ── Inference ────────────────────────────────────────────────────────


def detect_sample(sample):
    """Send a single sample to the monolith server."""
    t0 = time.perf_counter()

    try:
        if "raw_bytes" in sample:
            file_bytes = sample["raw_bytes"]
            filename = f"sample.{sample['mime'].split('/')[-1]}"
        elif "file_path" in sample:
            with open(sample["file_path"], "rb") as f:
                file_bytes = f.read()
            filename = Path(sample["file_path"]).name
        else:
            return {**sample, "error": "no data", "score": None}

        resp = requests.post(
            DETECT_URL,
            files={"file": (filename, io.BytesIO(file_bytes), sample["mime"])},
            timeout=600,
        )
        latency = time.perf_counter() - t0

        if resp.status_code == 200:
            body = resp.json()
            score = body.get("score")
            if score is None:
                # Try to get from confidence + verdict
                conf = body.get("confidence", 0.5)
                verdict = body.get("verdict", "real")
                score = conf if verdict == "fake" else (1.0 - conf)

            return {
                **sample,
                "score": float(score),
                "verdict": body.get("verdict"),
                "method": body.get("method", ""),
                "latency_s": latency,
                "error": None,
            }
        else:
            return {
                **sample,
                "score": None,
                "latency_s": latency,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
    except Exception as e:
        return {
            **sample,
            "score": None,
            "latency_s": time.perf_counter() - t0,
            "error": str(e)[:200],
        }


# ── Metrics ──────────────────────────────────────────────────────────


def compute_metrics(y_true, y_scores, threshold=0.5):
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    valid = ~np.isnan(y_scores)
    y_true = y_true[valid]
    y_scores = y_scores[valid]

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"n": int(len(y_true)), "error": "insufficient data"}

    y_pred = (y_scores >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "n": int(len(y_true)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc_roc": round(auc, 4) if auc is not None else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fpr": round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else None,
        "mean_fake_score": round(float(y_scores[y_true == 1].mean()), 4)
        if (y_true == 1).any() else None,
        "mean_real_score": round(float(y_scores[y_true == 0].mean()), 4)
        if (y_true == 0).any() else None,
    }


def compute_per_generator_metrics(results):
    """Compute metrics per generator."""
    by_gen = defaultdict(lambda: {"true": [], "scores": []})

    for r in results:
        if r.get("score") is None:
            continue
        gen = r["generator"]
        by_gen[gen]["true"].append(r["label"])
        by_gen[gen]["scores"].append(r["score"])

    per_gen = {}
    for gen, data in sorted(by_gen.items()):
        y_true = data["true"]
        y_scores = data["scores"]

        n = len(y_true)
        n_fake = sum(y_true)
        n_real = n - n_fake

        if n_fake > 0 and n_real > 0:
            metrics = compute_metrics(y_true, y_scores)
        elif n_fake > 0:
            # All fake -- report detection rate
            y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
            det_rate = sum(y_pred) / n
            metrics = {
                "n": n, "n_fake": n_fake, "n_real": 0,
                "detection_rate": round(det_rate, 4),
                "mean_score": round(float(np.mean(y_scores)), 4),
            }
        else:
            # All real -- report FPR
            y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
            fpr = sum(y_pred) / n
            metrics = {
                "n": n, "n_fake": 0, "n_real": n_real,
                "false_positive_rate": round(fpr, 4),
                "mean_score": round(float(np.mean(y_scores)), 4),
            }

        per_gen[gen] = metrics

    return per_gen


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Wild test evaluation against in-the-wild datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["defactify", "deepaction", "garystafford", "all"],
        help="Which dataset to evaluate",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per dataset (for quick testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Concurrent request workers",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: eval/results/wild_test_<dataset>.json)",
    )
    parser.add_argument(
        "--server-url", type=str, default=None,
        help="Monolith server URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    if args.server_url:
        global SERVER_URL, DETECT_URL
        SERVER_URL = args.server_url
        DETECT_URL = f"{SERVER_URL}/v1/detect"

    # Verify server is up
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=10).json()
        print(f"Server: {health['status']} | Models: {health['models_loaded']}")
        print(f"GPU: {health.get('gpu_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR: Server not responding at {SERVER_URL}: {e}")
        print("Start the monolith server first: python -m monolith.server")
        sys.exit(1)

    # Load samples
    datasets_to_run = (
        ["defactify", "deepaction", "garystafford"]
        if args.dataset == "all"
        else [args.dataset]
    )

    all_samples = []
    for ds_name in datasets_to_run:
        print(f"\nLoading {ds_name}...")
        if ds_name == "defactify":
            samples = load_defactify(args.max_samples)
        elif ds_name == "deepaction":
            samples = load_deepaction(args.max_samples)
        elif ds_name == "garystafford":
            samples = load_garystafford(args.max_samples)
        else:
            continue

        print(f"  {len(samples)} samples loaded")
        by_gen = defaultdict(int)
        for s in samples:
            by_gen[s["generator"]] += 1
        for g, c in sorted(by_gen.items()):
            label_info = "real" if g == "real" else "fake"
            print(f"    {g}: {c} ({label_info})")

        all_samples.extend(samples)

    if not all_samples:
        print("No samples loaded. Check dataset paths.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Total: {len(all_samples)} samples across {len(datasets_to_run)} datasets")
    print(f"{'=' * 60}\n")

    # Run inference
    results = []
    errors = 0

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(detect_sample, s): i
                for i, s in enumerate(all_samples)
            }
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                if r.get("error"):
                    errors += 1
                done = len(results)
                if done % 50 == 0 or done == len(all_samples):
                    print(
                        f"  Progress: {done}/{len(all_samples)} "
                        f"({errors} errors)"
                    )
    else:
        for i, sample in enumerate(all_samples):
            r = detect_sample(sample)
            results.append(r)
            if r.get("error"):
                errors += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(all_samples):
                print(
                    f"  Progress: {i + 1}/{len(all_samples)} "
                    f"({errors} errors)"
                )

    # Strip raw_bytes from results before saving (too large)
    for r in results:
        r.pop("raw_bytes", None)

    # Compute metrics
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    # Overall per dataset
    for ds_name in datasets_to_run:
        ds_results = [r for r in results if r.get("dataset") == ds_name]
        valid = [r for r in ds_results if r.get("score") is not None]

        if not valid:
            print(f"\n{ds_name}: No valid results")
            continue

        y_true = [r["label"] for r in valid]
        y_scores = [r["score"] for r in valid]

        overall = compute_metrics(y_true, y_scores)
        per_gen = compute_per_generator_metrics(valid)

        print(f"\n--- {ds_name.upper()} ({overall.get('n', 0)} samples) ---")
        if "auc_roc" in overall:
            print(
                f"  Overall AUC: {overall.get('auc_roc', 'N/A')} | "
                f"F1: {overall.get('f1', 'N/A')} | "
                f"Accuracy: {overall.get('accuracy', 'N/A')}"
            )
            print(
                f"  FPR: {overall.get('fpr', 'N/A')} | "
                f"Recall: {overall.get('recall', 'N/A')}"
            )

        print(f"\n  Per-generator:")
        for gen, metrics in per_gen.items():
            n = metrics.get("n", 0)
            if "auc_roc" in metrics:
                print(
                    f"    {gen:25s} n={n:4d}  "
                    f"AUC={metrics['auc_roc']:.4f}  "
                    f"F1={metrics['f1']:.4f}"
                )
            elif "detection_rate" in metrics:
                print(
                    f"    {gen:25s} n={n:4d}  "
                    f"DetRate={metrics['detection_rate']:.4f}  "
                    f"MeanScore={metrics['mean_score']:.4f}"
                )
            elif "false_positive_rate" in metrics:
                print(
                    f"    {gen:25s} n={n:4d}  "
                    f"FPR={metrics['false_positive_rate']:.4f}  "
                    f"MeanScore={metrics['mean_score']:.4f}"
                )

    # Save results
    output_path = args.output
    if not output_path:
        output_dir = PROJECT_ROOT / "eval" / "results"
        output_dir.mkdir(exist_ok=True)
        ds_suffix = args.dataset if args.dataset != "all" else "wild_all"
        output_path = str(
            output_dir / f"wild_test_{ds_suffix}.json"
        )

    # Build output payload
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "server_url": SERVER_URL,
        "datasets": datasets_to_run,
        "total_samples": len(all_samples),
        "total_errors": errors,
        "results_per_dataset": {},
    }

    for ds_name in datasets_to_run:
        ds_results = [r for r in results if r.get("dataset") == ds_name]
        valid = [r for r in ds_results if r.get("score") is not None]

        if not valid:
            continue

        y_true = [r["label"] for r in valid]
        y_scores = [r["score"] for r in valid]

        output_data["results_per_dataset"][ds_name] = {
            "overall": compute_metrics(y_true, y_scores),
            "per_generator": compute_per_generator_metrics(valid),
            "n_errors": len(ds_results) - len(valid),
        }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
