#!/usr/bin/env python3
"""Collect per-file predictions via the running monolith server API.

Unlike collect_predictions.py (which loads models in-process), this
script calls the running server's /v1/detect endpoint. This is simpler
and avoids GPU memory duplication.

Usage:
    # Server must be running: python -m monolith.server
    python eval/collect_predictions_api.py --modality image
    python eval/collect_predictions_api.py --modality audio
    python eval/collect_predictions_api.py --modality video
"""

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("collect_api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "master_eval"
METADATA_PATH = DATASET_DIR / "metadata.json"

SERVER_URL = os.getenv("DEEPSAFE_SERVER", "http://localhost:8000")
DETECT_URL = f"{SERVER_URL}/v1/detect"

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
    ".tiff": "image/tiff", ".tif": "image/tiff",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".m4a": "audio/mp4",
    ".mp4": "video/mp4", ".avi": "video/x-msvideo",
    ".mov": "video/quicktime", ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}

_MODALITY_MAP = {"images": "image", "image": "image", "audio": "audio", "video": "video"}


@dataclass
class Sample:
    id: str
    path: str
    modality: str
    label: str
    generator: str
    format: str


def load_metadata(modality_filter: Optional[str] = None) -> List[Sample]:
    with open(METADATA_PATH) as f:
        raw = json.load(f)
    samples = []
    for entry in raw:
        mod = _MODALITY_MAP.get(entry["modality"], entry["modality"])
        if modality_filter and mod != modality_filter:
            continue
        file_path = DATASET_DIR / entry["path"]
        if not file_path.exists():
            continue
        samples.append(Sample(
            id=entry["id"], path=entry["path"],
            modality=mod, label=entry["label"],
            generator=entry.get("generator", "unknown"),
            format=entry.get("format", ""),
        ))
    return samples


def detect_one(sample: Sample, session: requests.Session) -> dict:
    """Call server /v1/detect for a single file."""
    file_path = DATASET_DIR / sample.path
    ext = file_path.suffix.lower()
    mime = MIME_MAP.get(ext, "application/octet-stream")

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, mime)}
        r = session.post(DETECT_URL, files=files, timeout=300)

    if r.status_code != 200:
        return {"_error": f"HTTP {r.status_code}: {r.text[:200]}"}

    return r.json()


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=["image", "audio", "video"])
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent requests (keep low for GPU)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    args = parser.parse_args()

    # Verify server is up
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=60).json()
        logger.info("Server: %s, %d models loaded", health["status"], health["models_loaded"])
    except Exception as e:
        logger.error("Server not reachable at %s: %s", SERVER_URL, e)
        sys.exit(1)

    # Load metadata
    samples = load_metadata(args.modality)
    if args.limit:
        samples = samples[:args.limit]
    logger.info("Loaded %d %s samples", len(samples), args.modality)

    # Output path
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_path = Path(
        args.output or str(
            PROJECT_ROOT / "eval" / "results"
            / f"{timestamp}-{args.modality}-predictions.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        logger.info("Resuming from checkpoint: %d done", len(completed))

    # Collect predictions
    session = requests.Session()
    start_time = time.time()
    errors = 0

    for i, sample in enumerate(samples):
        if sample.id in completed:
            continue

        try:
            result = detect_one(sample, session)
        except Exception as e:
            logger.warning("[%d/%d] %s: request error: %s", i+1, len(samples), sample.id, e)
            errors += 1
            continue

        if "_error" in result:
            logger.warning("[%d/%d] %s: %s", i+1, len(samples), sample.id, result["_error"])
            errors += 1
            continue

        # Extract model probabilities
        model_probs = {}
        model_details = {}
        for model_name, mresult in result.get("model_results", {}).items():
            prob = mresult.get("probability")
            if prob is not None:
                model_probs[model_name] = float(prob)
            model_details[model_name] = mresult

        completed[sample.id] = {
            "id": sample.id,
            "path": sample.path,
            "modality": sample.modality,
            "label": sample.label,
            "label_int": 1 if sample.label == "fake" else 0,
            "generator": sample.generator,
            "model_probs": model_probs,
            "model_details": model_details,
            "ensemble_verdict": result.get("verdict"),
            "ensemble_score": result.get("score"),
            "ensemble_method": result.get("method"),
        }

        done = len(completed)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            logger.info(
                "[%d/%d] %s → %s (%.3f), %d models, %.1f/s",
                i+1, len(samples), sample.id,
                result.get("verdict", "?"), result.get("score", 0),
                len(model_probs), rate,
            )

        if done % args.checkpoint_interval == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f)

    elapsed = time.time() - start_time

    # Save
    output = {
        "metadata": {
            "timestamp": timestamp,
            "modality": args.modality,
            "total_samples": len(samples),
            "completed": len(completed),
            "errors": errors,
            "elapsed_seconds": round(elapsed, 1),
            "server_url": SERVER_URL,
        },
        "predictions": completed,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    logger.info(
        "Done: %d/%d predictions saved to %s (%.0fs, %d errors)",
        len(completed), len(samples), output_path, elapsed, errors,
    )

    # Summary
    labels = Counter(v["label"] for v in completed.values())
    verdicts = Counter(v.get("ensemble_verdict", "?") for v in completed.values())
    logger.info("Labels: %s", dict(labels))
    logger.info("Verdicts: %s", dict(verdicts))


if __name__ == "__main__":
    main()
