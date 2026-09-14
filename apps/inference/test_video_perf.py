"""Video inference performance test.

Tests each video model individually for correctness and speed,
then runs all models in parallel to measure end-to-end throughput.

Usage:
    cd /workspace/deepsafe
    python -m monolith.test_video_perf
"""

import gc
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("video_perf")

# ── Config ──────────────────────────────────────────────────────────────────

VIDEO_MODELS = [
    "fakestormer",
    "sbi",
    "dfd_fcg",
    "pwtf_dvd",
    "lipfd",
    "recce",
    "mintime",
]

SAMPLE_VIDEO = Path("tests/assets/sample_video.mp4")
DATASET_DIR = Path("dataset/master_eval_small/video")

WARMUP_RUNS = 1
TIMED_RUNS = 3


def _find_dataset_videos(max_files: int = 10) -> list[Path]:
    """Gather a small set of eval dataset videos."""
    if not DATASET_DIR.exists():
        return []
    videos = sorted(DATASET_DIR.rglob("*.mp4"))[:max_files]
    return videos


def _load_model(name: str, device: torch.device):
    """Load a single model and return (predictor, load_time_ms)."""
    from config import get_weights_path

    from models import get_predictor_class

    cls = get_predictor_class(name)
    predictor = cls()
    weights_dir = get_weights_path(name)
    start = time.perf_counter()
    predictor.load(weights_dir, device)
    load_ms = (time.perf_counter() - start) * 1000
    return predictor, load_ms


def _run_inference(predictor, video_bytes: bytes) -> dict:
    """Run inference and return result dict."""
    return predictor.predict(video_bytes)


def test_single_model(
    name: str,
    device: torch.device,
    video_bytes: bytes,
) -> dict:
    """Test a single model: load, warmup, timed runs.

    Returns dict with timings and results.
    """
    separator = "=" * 60
    logger.info(separator)
    logger.info("TESTING: %s", name.upper())
    logger.info(separator)

    # Load
    try:
        predictor, load_ms = _load_model(name, device)
        logger.info("  Loaded in %.0fms", load_ms)
    except Exception as e:
        logger.error("  LOAD FAILED: %s", e, exc_info=True)
        return {
            "model": name,
            "status": "LOAD_FAILED",
            "error": str(e),
        }

    # Health
    health = predictor.health()
    logger.info("  Device: %s", health.get("device"))

    # Warmup
    logger.info("  Warmup (%d runs)...", WARMUP_RUNS)
    for i in range(WARMUP_RUNS):
        try:
            result = _run_inference(predictor, video_bytes)
            prob = result.get("probability")
            lat = result.get("latency_ms", 0)
            logger.info(
                "    warmup %d: prob=%.4f, latency=%.1fms",
                i + 1,
                prob if prob is not None else -1,
                lat,
            )
            if prob is None:
                logger.error("    ERROR: %s", result.get("error"))
                return {
                    "model": name,
                    "status": "INFERENCE_FAILED",
                    "error": result.get("error"),
                    "load_ms": load_ms,
                }
        except Exception as e:
            logger.error("    warmup EXCEPTION: %s", e, exc_info=True)
            return {
                "model": name,
                "status": "INFERENCE_FAILED",
                "error": str(e),
                "load_ms": load_ms,
            }

    # Timed runs
    logger.info("  Timed runs (%d)...", TIMED_RUNS)
    latencies = []
    probabilities = []
    for i in range(TIMED_RUNS):
        start = time.perf_counter()
        result = _run_inference(predictor, video_bytes)
        wall_ms = (time.perf_counter() - start) * 1000
        prob = result.get("probability")
        lat = result.get("latency_ms", wall_ms)
        latencies.append(lat)
        probabilities.append(prob)
        logger.info(
            "    run %d: prob=%.4f, latency=%.1fms (wall=%.1fms)",
            i + 1,
            prob if prob is not None else -1,
            lat,
            wall_ms,
        )

    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    avg_prob = sum(p for p in probabilities if p is not None) / max(
        sum(1 for p in probabilities if p is not None), 1
    )

    # Validate output
    valid = all(p is not None and 0.0 <= p <= 1.0 for p in probabilities)
    prob_spread = (
        (
            max(p for p in probabilities if p is not None)
            - min(p for p in probabilities if p is not None)
        )
        if valid
        else -1
    )

    logger.info("  RESULTS:")
    logger.info("    Avg latency : %.1fms", avg_lat)
    logger.info("    Min latency : %.1fms", min_lat)
    logger.info("    Max latency : %.1fms", max_lat)
    logger.info("    Avg prob    : %.4f", avg_prob)
    logger.info("    Prob spread : %.6f", prob_spread)
    logger.info("    Valid output: %s", valid)

    status = "PASS" if valid else "FAIL"
    if prob_spread > 0.01:
        logger.warning(
            "    WARNING: High probability variance (%.6f)",
            prob_spread,
        )

    logger.info("  STATUS: %s", status)

    # Cleanup
    del predictor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": name,
        "status": status,
        "load_ms": round(load_ms, 1),
        "avg_latency_ms": round(avg_lat, 1),
        "min_latency_ms": round(min_lat, 1),
        "max_latency_ms": round(max_lat, 1),
        "avg_probability": round(avg_prob, 4),
        "prob_spread": round(prob_spread, 6),
        "valid": valid,
    }


def test_all_parallel(
    device: torch.device,
    dataset_videos: list[Path],
) -> dict:
    """Load all video models and run them in parallel on the dataset.

    Returns per-file average inference time across all models.
    """
    separator = "=" * 60
    logger.info(separator)
    logger.info("PARALLEL TEST: All video models on %d files", len(dataset_videos))
    logger.info(separator)

    from config import get_weights_path

    from models import get_predictor_class

    # Load all models
    models = {}
    total_load_ms = 0
    for name in VIDEO_MODELS:
        try:
            predictor, load_ms = _load_model(name, device)
            models[name] = predictor
            total_load_ms += load_ms
            logger.info("  Loaded %s (%.0fms)", name, load_ms)
        except Exception as e:
            logger.error("  SKIP %s: %s", name, e)

    logger.info(
        "  %d/%d models loaded (total %.1fs)",
        len(models),
        len(VIDEO_MODELS),
        total_load_ms / 1000,
    )

    if not models:
        logger.error("No models loaded - cannot run parallel test")
        return {}

    # Report VRAM
    if device.type == "cuda":
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("  VRAM: %.1f / %.1f GB", vram_used, vram_total)

    # Warmup all models with first video
    warmup_bytes = dataset_videos[0].read_bytes()
    logger.info("  Warming up all models...")
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(m.predict, warmup_bytes): n for n, m in models.items()
        }
        for f in as_completed(futures):
            name = futures[f]
            try:
                r = f.result()
                logger.info(
                    "    %s warmup: prob=%.4f, %.1fms",
                    name,
                    r.get("probability", -1) or -1,
                    r.get("latency_ms", 0),
                )
            except Exception as e:
                logger.error("    %s warmup failed: %s", name, e)

    # Run timed inference on all dataset videos
    logger.info("  Running timed inference on %d videos...", len(dataset_videos))
    per_file_results = []

    for video_path in dataset_videos:
        video_bytes = video_path.read_bytes()
        file_start = time.perf_counter()

        # Fan out all models in parallel
        model_times = {}
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {
                executor.submit(m.predict, video_bytes): n for n, m in models.items()
            }
            for f in as_completed(futures):
                name = futures[f]
                try:
                    r = f.result()
                    model_times[name] = r.get("latency_ms", 0)
                except Exception as e:
                    logger.error("    %s error on %s: %s", name, video_path.name, e)
                    model_times[name] = -1

        file_wall_ms = (time.perf_counter() - file_start) * 1000
        slowest_model = max(model_times, key=model_times.get)
        slowest_ms = model_times[slowest_model]

        per_file_results.append(
            {
                "file": video_path.name,
                "wall_ms": round(file_wall_ms, 1),
                "slowest_model": slowest_model,
                "slowest_ms": round(slowest_ms, 1),
                "model_times": {k: round(v, 1) for k, v in model_times.items()},
            }
        )

        logger.info(
            "    %s: wall=%.0fms, slowest=%s (%.0fms)",
            video_path.name,
            file_wall_ms,
            slowest_model,
            slowest_ms,
        )

    # Summary
    avg_wall = sum(r["wall_ms"] for r in per_file_results) / len(per_file_results)
    avg_slowest = sum(r["slowest_ms"] for r in per_file_results) / len(per_file_results)

    per_model_avg = {}
    for name in models:
        times = [
            r["model_times"].get(name, 0)
            for r in per_file_results
            if r["model_times"].get(name, -1) > 0
        ]
        if times:
            per_model_avg[name] = round(sum(times) / len(times), 1)

    logger.info("-" * 60)
    logger.info(
        "PARALLEL RESULTS (%d files, %d models):", len(dataset_videos), len(models)
    )
    logger.info("  Avg wall time per file   : %.1fms", avg_wall)
    logger.info("  Avg slowest model per file: %.1fms", avg_slowest)
    logger.info("  Per-model avg latency:")
    for name, avg in sorted(per_model_avg.items(), key=lambda x: x[1]):
        logger.info("    %-12s: %.1fms", name, avg)

    # Cleanup
    del models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "num_files": len(dataset_videos),
        "num_models": len(VIDEO_MODELS),
        "avg_wall_ms": round(avg_wall, 1),
        "avg_slowest_ms": round(avg_slowest, 1),
        "per_model_avg_ms": per_model_avg,
        "per_file": per_file_results,
    }


def main():
    from models.base import get_device, setup_inference_optimizations

    device = get_device()
    setup_inference_optimizations(device)

    if not SAMPLE_VIDEO.exists():
        logger.error("Sample video not found: %s", SAMPLE_VIDEO)
        sys.exit(1)

    video_bytes = SAMPLE_VIDEO.read_bytes()
    logger.info(
        "Test video: %s (%d KB)",
        SAMPLE_VIDEO.name,
        len(video_bytes) // 1024,
    )

    # Phase 1: Test each model individually
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Individual model testing")
    logger.info("=" * 60)

    individual_results = {}
    for name in VIDEO_MODELS:
        result = test_single_model(name, device, video_bytes)
        individual_results[name] = result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Phase 1 summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 SUMMARY")
    logger.info("=" * 60)
    logger.info(
        "%-14s %-10s %10s %10s %10s",
        "Model",
        "Status",
        "Avg(ms)",
        "Min(ms)",
        "Prob",
    )
    logger.info("-" * 60)
    for name, r in individual_results.items():
        logger.info(
            "%-14s %-10s %10s %10s %10s",
            name,
            r.get("status", "?"),
            r.get("avg_latency_ms", "?"),
            r.get("min_latency_ms", "?"),
            r.get("avg_probability", "?"),
        )

    # Check if any failed
    failed = [n for n, r in individual_results.items() if r.get("status") != "PASS"]
    if failed:
        logger.error("FAILED MODELS: %s", ", ".join(failed))
        logger.error("Fix these before running Phase 2.")
        sys.exit(1)

    # Phase 2: All models in parallel on dataset
    dataset_videos = _find_dataset_videos(max_files=10)
    if not dataset_videos:
        logger.warning("No dataset videos found. Skipping Phase 2.")
        sys.exit(0)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: All models in parallel on dataset")
    logger.info("=" * 60)

    parallel_results = test_all_parallel(device, dataset_videos)

    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
