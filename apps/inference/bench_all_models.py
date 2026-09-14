"""Benchmark all 22 models hot-loaded, parallel inference per modality.

For each test file, runs the appropriate modality models in parallel
(image→7 image models, video→7 video models, audio→4 audio models)
plus provenance models matched to their supported media type.

Usage:
    DEEPSAFE_DEVICE=cuda python -m monolith.bench_all_models
"""

import gc
import logging
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bench")

from config import MODEL_REGISTRY, get_weights_path
from models.base import get_device, setup_inference_optimizations

from models import get_predictor_class

# Which provenance models apply to which media types
_PROV_MEDIA_MAP = {
    "c2pa": ["image", "audio", "video"],
    "sdxl_watermark": ["image"],
    "audioseal": ["audio"],
    "videoseal": ["image", "video"],
}


def get_models_for_file(media_type: str, loaded: dict) -> list:
    """Return list of model names to run for a given media type."""
    names = []
    for name, model in loaded.items():
        modality = MODEL_REGISTRY[name].modality
        if modality == media_type:
            names.append(name)
        elif modality == "provenance":
            if media_type in _PROV_MEDIA_MAP.get(name, []):
                names.append(name)
    return names


def detect_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"):
        return "image"
    elif ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        return "video"
    elif ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        return "audio"
    return "image"


def main():
    device = get_device()
    setup_inference_optimizations(device)

    # ── Load ALL models ──────────────────────────────────────────
    logger.info("Loading all %d models...", len(MODEL_REGISTRY))
    loaded = {}
    load_start = time.perf_counter()
    for name in MODEL_REGISTRY:
        try:
            cls = get_predictor_class(name)
            p = cls()
            t0 = time.perf_counter()
            p.load(get_weights_path(name), device)
            ms = (time.perf_counter() - t0) * 1000
            loaded[name] = p
            logger.info(
                "  %-16s loaded (%.0fms)",
                name,
                ms,
            )
        except Exception as e:
            logger.error("  %-16s FAILED: %s", name, str(e)[:80])
        gc.collect()
        torch.cuda.empty_cache()

    total_load = (time.perf_counter() - load_start) * 1000
    logger.info(
        "%d/%d loaded in %.1fs",
        len(loaded),
        len(MODEL_REGISTRY),
        total_load / 1000,
    )
    if device.type == "cuda":
        vram = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("VRAM: %.2f / %.1f GB", vram, vram_total)

    # ── Collect test files ───────────────────────────────────────
    dataset = Path("dataset/master_eval_small")
    test_files = []

    for subdir in ["images", "audio", "video"]:
        d = dataset / subdir
        if not d.exists():
            continue
        files = sorted(d.rglob("*"))
        files = [
            f
            for f in files
            if f.is_file()
            and f.suffix.lower()
            in (
                ".jpg",
                ".jpeg",
                ".png",
                ".mp4",
                ".avi",
                ".wav",
                ".mp3",
                ".flac",
            )
        ]
        test_files.extend(files[:5])  # 5 per modality

    logger.info("Test files: %d", len(test_files))

    # ── Warmup ───────────────────────────────────────────────────
    logger.info("Warming up...")
    warmup_data = {
        "image": Path("tests/assets/sample_image.jpg").read_bytes(),
        "audio": Path("tests/assets/sample_audio.wav").read_bytes(),
        "video": Path("tests/assets/sample_video.mp4").read_bytes(),
    }
    for name, model in loaded.items():
        modality = MODEL_REGISTRY[name].modality
        if modality == "provenance":
            media = _PROV_MEDIA_MAP.get(name, ["image"])[0]
        else:
            media = modality
        try:
            model.predict(warmup_data.get(media, warmup_data["image"]))
        except Exception:
            pass

    # ── Benchmark ────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 80)
    logger.info("BENCHMARK: parallel inference per modality")
    logger.info("=" * 80)

    per_file_results = []

    for fpath in test_files:
        media_type = detect_media_type(fpath)
        model_names = get_models_for_file(media_type, loaded)
        if not model_names:
            continue

        raw_bytes = fpath.read_bytes()
        label = "FAKE" if "/fake/" in str(fpath) else "REAL"

        t0 = time.perf_counter()
        model_times = {}
        model_probs = {}

        with ThreadPoolExecutor(max_workers=len(model_names)) as ex:
            futures = {ex.submit(loaded[n].predict, raw_bytes): n for n in model_names}
            for f in as_completed(futures, timeout=300):
                n = futures[f]
                try:
                    r = f.result()
                    model_times[n] = r.get("latency_ms", 0)
                    model_probs[n] = r.get("probability")
                except Exception as e:
                    model_times[n] = -1
                    model_probs[n] = None

        wall_ms = (time.perf_counter() - t0) * 1000
        slowest = max(model_times, key=model_times.get) if model_times else "?"

        per_file_results.append(
            {
                "file": fpath.name,
                "media": media_type,
                "label": label,
                "wall_ms": wall_ms,
                "n_models": len(model_names),
                "slowest": slowest,
                "slowest_ms": model_times.get(slowest, 0),
                "times": model_times,
                "probs": model_probs,
            }
        )

        logger.info(
            "  %-25s [%s %s] %2d models  wall=%7.0fms  " "slowest=%s(%.0fms)",
            fpath.name,
            media_type[:3],
            label,
            len(model_names),
            wall_ms,
            slowest,
            model_times.get(slowest, 0),
        )

    # ── Summary ──────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    by_modality = defaultdict(list)
    for r in per_file_results:
        by_modality[r["media"]].append(r)

    for media in ["image", "audio", "video"]:
        files = by_modality.get(media, [])
        if not files:
            continue
        walls = [r["wall_ms"] for r in files]
        logger.info("")
        logger.info(
            "%s (%d files, %d models per file):",
            media.upper(),
            len(files),
            files[0]["n_models"] if files else 0,
        )
        logger.info(
            "  Wall time: avg=%.0fms  min=%.0fms  max=%.0fms",
            np.mean(walls),
            np.min(walls),
            np.max(walls),
        )

        # Per-model breakdown
        all_model_names = set()
        for r in files:
            all_model_names.update(r["times"].keys())
        for name in sorted(all_model_names):
            times = [
                r["times"][name]
                for r in files
                if name in r["times"] and r["times"][name] > 0
            ]
            if times:
                logger.info(
                    "    %-16s: avg=%7.0fms  min=%7.0fms  max=%7.0fms",
                    name,
                    np.mean(times),
                    np.min(times),
                    np.max(times),
                )

    # Grand totals
    all_walls = [r["wall_ms"] for r in per_file_results]
    logger.info("")
    logger.info(
        "OVERALL (%d files across all modalities):",
        len(per_file_results),
    )
    logger.info(
        "  Wall time: avg=%.0fms  min=%.0fms  max=%.0fms  total=%.1fs",
        np.mean(all_walls),
        np.min(all_walls),
        np.max(all_walls),
        sum(all_walls) / 1000,
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
