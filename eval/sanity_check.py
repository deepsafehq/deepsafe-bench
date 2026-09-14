#!/usr/bin/env python3
"""Sanity check: validate all models produce correct directional output.

Tests a small sample of known real + known fake media from master_eval.
Catches models that always return fake, always return real, or return
garbage before wasting hours on full inference.

Usage:
    python eval/sanity_check.py               # via running server
    python eval/sanity_check.py --samples 10  # more samples per class
"""

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "master_eval"
METADATA_PATH = DATASET_DIR / "metadata.json"
SERVER_URL = os.getenv("DEEPSAFE_SERVER", "http://localhost:8000")

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".mp4": "video/mp4", ".avi": "video/x-msvideo", ".mov": "video/quicktime",
}

_MODALITY_MAP = {"images": "image", "audio": "audio", "video": "video"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5, help="Samples per class per modality")
    args = parser.parse_args()

    # Check server
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=30).json()
        print(f"Server: {health['status']}, {health['models_loaded']} models")
    except Exception as e:
        print(f"ERROR: Server not reachable: {e}")
        sys.exit(1)

    # Load metadata
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    # Group by modality+label, pick random samples
    groups = defaultdict(list)
    for entry in meta:
        mod = _MODALITY_MAP.get(entry["modality"], entry["modality"])
        path = DATASET_DIR / entry["path"]
        if path.exists():
            groups[(mod, entry["label"])].append(entry)

    random.seed(42)
    test_plan = []
    for (mod, label), entries in sorted(groups.items()):
        sample = random.sample(entries, min(args.samples, len(entries)))
        for e in sample:
            test_plan.append((mod, label, e))

    print(f"\nSanity check: {len(test_plan)} samples "
          f"({args.samples} real + {args.samples} fake per modality)\n")

    # Run inference
    session = requests.Session()
    results_by_model = defaultdict(lambda: {"real_probs": [], "fake_probs": []})
    modality_results = defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0})

    for mod, label, entry in test_plan:
        path = DATASET_DIR / entry["path"]
        ext = path.suffix.lower()
        mime = MIME_MAP.get(ext, "application/octet-stream")

        try:
            with open(path, "rb") as f:
                r = session.post(
                    f"{SERVER_URL}/v1/detect",
                    files={"file": (path.name, f, mime)},
                    timeout=120,
                )
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}: {entry['id']} ({mod}/{label})")
                modality_results[mod]["errors"] += 1
                continue

            data = r.json()
            verdict = data.get("verdict", "?")
            score = data.get("score", 0.5)

            # Track per-model probabilities
            for model_name, mresult in data.get("model_results", {}).items():
                prob = mresult.get("probability")
                if prob is not None:
                    key = "fake_probs" if label == "fake" else "real_probs"
                    results_by_model[model_name][key].append(prob)

            # Track ensemble accuracy
            expected = label
            modality_results[mod]["total"] += 1
            if verdict == expected:
                modality_results[mod]["correct"] += 1

        except Exception as e:
            print(f"  ERROR: {entry['id']} ({mod}/{label}): {e}")
            modality_results[mod]["errors"] += 1

    # Report: ensemble accuracy per modality
    print("\n" + "=" * 70)
    print("ENSEMBLE ACCURACY (sanity check)")
    print("=" * 70)
    all_ok = True
    for mod in ["image", "audio", "video"]:
        r = modality_results[mod]
        if r["total"] == 0:
            print(f"  {mod}: NO DATA")
            continue
        acc = r["correct"] / r["total"]
        status = "OK" if acc >= 0.5 else "BAD"
        if acc < 0.5:
            all_ok = False
        print(f"  {mod}: {r['correct']}/{r['total']} correct ({acc:.0%}) "
              f"[{r['errors']} errors] {status}")

    # Report: per-model directionality
    print("\n" + "=" * 70)
    print("PER-MODEL DIRECTIONALITY")
    print(f"{'Model':<20} {'Avg(real)':<12} {'Avg(fake)':<12} {'Direction':<12} {'Status'}")
    print("-" * 70)

    model_issues = []
    for model_name in sorted(results_by_model.keys()):
        d = results_by_model[model_name]
        real_probs = d["real_probs"]
        fake_probs = d["fake_probs"]

        avg_real = sum(real_probs) / len(real_probs) if real_probs else float("nan")
        avg_fake = sum(fake_probs) / len(fake_probs) if fake_probs else float("nan")

        if fake_probs and real_probs:
            if avg_fake > avg_real:
                direction = "CORRECT"
                status = "OK"
            elif abs(avg_fake - avg_real) < 0.05:
                direction = "FLAT"
                status = "WARN"
                model_issues.append((model_name, "flat — no discrimination"))
            else:
                direction = "INVERTED"
                status = "BAD"
                model_issues.append((model_name, "inverted — higher on real"))
                all_ok = False
        else:
            direction = "?"
            status = "SKIP"

        print(f"  {model_name:<18} {avg_real:<12.4f} {avg_fake:<12.4f} "
              f"{direction:<12} {status}")

    # Summary
    print("\n" + "=" * 70)
    if model_issues:
        print(f"ISSUES FOUND ({len(model_issues)}):")
        for name, issue in model_issues:
            print(f"  {name}: {issue}")
    if all_ok:
        print("ALL CHECKS PASSED — safe to run full experiment")
    else:
        print("SOME CHECKS FAILED — fix issues before full experiment")
        sys.exit(1)


if __name__ == "__main__":
    main()
