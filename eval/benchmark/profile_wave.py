#!/usr/bin/env python3
"""VRAM and latency profiler for modality waves.

Profiles a single modality wave (image, audio, or video): builds
containers, measures per-model VRAM consumption, records cold start
time, then runs latency benchmarks at multiple input sizes.

Usage:
    python profile_wave.py --modality {image,audio,video}

The script runs on the GPU host machine and communicates with model
containers via localhost-mapped ports. Docker Compose files are
expected at the project root.
"""

import argparse
import base64
import io
import json
import math
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

# Sibling module imports.
sys.path.insert(0, str(Path(__file__).parent))
from gpu_utils import (
    docker_stats,
    gpu_stats,
    wait_for_health,
)
from models import ModelDef, get_docker_services, get_models_by_modality

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Constants ----

COMPOSE_FILE = str(_PROJECT_ROOT / "docker-compose.yml")
COMPOSE_PROD = str(_PROJECT_ROOT / "docker-compose.prod.yml")
COMPOSE_CWD = str(_PROJECT_ROOT)
RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"

BUILD_TIMEOUT_S = 1800
START_TIMEOUT_S = 600
INTER_MODEL_SLEEP_S = 3

WARMUP_RUNS = 3
TIMED_RUNS = 20
REQUEST_TIMEOUT_S = 300

# Payload sizes per modality.
IMAGE_SIZES = [(512, 512), (1024, 1024), (2048, 2048)]
AUDIO_DURATIONS_S = [1, 5, 15, 30]
VIDEO_DURATIONS_S = [2, 5, 10]

SAMPLE_RATE = 16000  # 16 kHz mono for audio payloads.


# ---- Payload Generation ----


def _generate_image_payload(width: int, height: int) -> str:
    """Generate a random RGB JPEG image and return base64-encoded bytes.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Base64-encoded JPEG string.
    """
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _generate_wav_payload(duration_s: int) -> str:
    """Generate a random float32 WAV with a proper RIFF header.

    Args:
        duration_s: Duration of the audio in seconds.

    Returns:
        Base64-encoded WAV string.
    """
    n_samples = SAMPLE_RATE * duration_s
    rng = np.random.RandomState(42)
    samples = rng.uniform(-1.0, 1.0, n_samples).astype(np.float32)

    # Build RIFF/WAV header for PCM float32.
    n_channels = 1
    bits_per_sample = 32
    byte_rate = SAMPLE_RATE * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = n_samples * block_align
    # WAVE_FORMAT_IEEE_FLOAT = 3
    audio_format = 3

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,  # File size - 8
        b"WAVE",
        b"fmt ",
        16,  # Subchunk1Size (PCM)
        audio_format,
        n_channels,
        SAMPLE_RATE,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    wav_bytes = header + samples.tobytes()
    return base64.b64encode(wav_bytes).decode()


def _generate_video_payload(duration_s: int) -> str:
    """Generate a synthetic MP4 video using ffmpeg testsrc.

    Args:
        duration_s: Duration of the video in seconds.

    Returns:
        Base64-encoded MP4 string.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration_s}:size=640x480:rate=25",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-f", "mp4",
            tmp_path,
        ]
        subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            check=True,
        )
        data = Path(tmp_path).read_bytes()
        return base64.b64encode(data).decode()
    finally:
        p = Path(tmp_path)
        if p.exists():
            p.unlink()


def generate_payloads(modality: str) -> list[tuple[str, str]]:
    """Generate synthetic test payloads for the given modality.

    Args:
        modality: One of 'image', 'audio', or 'video'.

    Returns:
        List of (label, base64_data) tuples.
    """
    payloads = []
    if modality == "image":
        for w, h in IMAGE_SIZES:
            label = f"{w}x{h}"
            print(f"  Generating image payload: {label}", flush=True)
            payloads.append((label, _generate_image_payload(w, h)))
    elif modality == "audio":
        for dur in AUDIO_DURATIONS_S:
            label = f"{dur}s"
            print(f"  Generating audio payload: {label}", flush=True)
            payloads.append((label, _generate_wav_payload(dur)))
    elif modality == "video":
        for dur in VIDEO_DURATIONS_S:
            label = f"{dur}s"
            print(f"  Generating video payload: {label}", flush=True)
            payloads.append((label, _generate_video_payload(dur)))
    else:
        raise ValueError(f"Unknown modality: {modality}")

    print(f"  Generated {len(payloads)} payloads.", flush=True)
    return payloads


# ---- Docker Compose Helpers ----


def _compose_cmd(*args: str) -> list[str]:
    """Build a docker compose command with both compose files.

    Args:
        *args: Additional arguments for docker compose.

    Returns:
        Full command list.
    """
    return [
        "docker", "compose",
        "-f", COMPOSE_FILE,
        "-f", COMPOSE_PROD,
        *args,
    ]


def build_services(services: list[str]) -> dict[str, bool]:
    """Build Docker containers one by one, tolerating individual failures.

    Args:
        services: List of docker-compose service names to build.

    Returns:
        Dict mapping service name to build success (True/False).
    """
    results = {}
    for svc in services:
        cmd = _compose_cmd("build", svc)
        print(f"  Building {svc}...", flush=True)
        try:
            subprocess.run(
                cmd,
                cwd=COMPOSE_CWD,
                timeout=BUILD_TIMEOUT_S,
                check=True,
                capture_output=True,
            )
            results[svc] = True
            print(f"    {svc}: OK", flush=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            results[svc] = False
            err_msg = ""
            if hasattr(exc, "stderr") and exc.stderr:
                err_msg = exc.stderr[-300:] if isinstance(exc.stderr, str) else exc.stderr.decode(errors="replace")[-300:]
            print(f"    {svc}: FAILED -- {err_msg}", flush=True)
    succeeded = sum(1 for v in results.values() if v)
    print(f"  Build complete: {succeeded}/{len(services)} succeeded.", flush=True)
    return results


def start_services(services: list[str]) -> None:
    """Start Docker containers for the given service names.

    Args:
        services: List of docker-compose service names to start.
    """
    cmd = _compose_cmd("up", "-d", *services)
    print(f"  Starting services: {', '.join(services)}", flush=True)
    subprocess.run(
        cmd,
        cwd=COMPOSE_CWD,
        timeout=START_TIMEOUT_S,
        check=True,
    )


def stop_services(services: list[str]) -> None:
    """Stop Docker containers for the given service names.

    Args:
        services: List of docker-compose service names to stop.
    """
    cmd = _compose_cmd("stop", *services)
    print(f"  Stopping services: {', '.join(services)}", flush=True)
    subprocess.run(
        cmd,
        cwd=COMPOSE_CWD,
        timeout=120,
        check=True,
    )


# ---- Percentile Helpers ----


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile from a sorted list.

    Uses linear interpolation between closest ranks.

    Args:
        sorted_values: A sorted list of numeric values.
        p: Percentile in [0, 100].

    Returns:
        Interpolated percentile value.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _compute_stdev(values: list[float]) -> float:
    """Compute sample standard deviation.

    Args:
        values: List of numeric values.

    Returns:
        Sample standard deviation, or 0.0 if fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


# ---- Profiling Phases ----


def _payload_key_for_modality(modality: str) -> str:
    """Return the JSON payload key for the given modality.

    Args:
        modality: One of 'image', 'audio', or 'video'.

    Returns:
        Payload key string (e.g. 'image_data').
    """
    key_map = {
        "image": "image_data",
        "audio": "audio_data",
        "video": "video_data",
    }
    return key_map[modality]


def profile_individual_vram(
    models: list[ModelDef],
    services: list[str],
) -> list[dict[str, Any]]:
    """Profile VRAM for each model individually.

    Starts each model ALONE, measures VRAM delta (loaded minus baseline),
    CPU RAM, power draw, and cold start time, then stops the container.

    Args:
        models: List of ModelDef objects for the wave.
        services: List of corresponding docker-compose service names.

    Returns:
        List of per-model VRAM profile dicts.
    """
    results = []
    for model, service in zip(models, services):
        print(
            f"\n  --- Individual VRAM: {model.name} "
            f"(service={service}) ---",
            flush=True,
        )

        # Record baseline GPU state before starting.
        baseline_gpu = gpu_stats()
        baseline_vram = (
            baseline_gpu.get("vram_used_mb", 0.0)
            if baseline_gpu
            else 0.0
        )

        # Start the single service and time until healthy.
        t_start = time.perf_counter()
        start_services([service])

        healthy, _ = wait_for_health(
            model.port,
            timeout=START_TIMEOUT_S,
        )
        cold_start_s = time.perf_counter() - t_start

        if not healthy:
            print(
                f"    WARNING: {model.name} did not become healthy "
                f"within {START_TIMEOUT_S}s.",
                flush=True,
            )

        # Measure loaded state.
        loaded_gpu = gpu_stats()
        loaded_vram = (
            loaded_gpu.get("vram_used_mb", 0.0)
            if loaded_gpu
            else 0.0
        )
        vram_delta_mb = loaded_vram - baseline_vram

        container_stats = docker_stats(model.container_name)
        power_w = loaded_gpu.get("power_draw_w", 0.0) if loaded_gpu else 0.0

        entry = {
            "model": model.name,
            "service": service,
            "container": model.container_name,
            "port": model.port,
            "healthy": healthy,
            "cold_start_s": round(cold_start_s, 2),
            "vram_delta_mb": round(vram_delta_mb, 1),
            "cpu_ram_mb": container_stats.get("mem_usage_mb"),
            "power_draw_w": round(power_w, 1) if power_w else None,
            "gpu_stats_loaded": loaded_gpu,
        }
        results.append(entry)

        print(
            f"    cold_start={cold_start_s:.1f}s  "
            f"vram_delta={vram_delta_mb:.1f}MB  "
            f"cpu_ram={container_stats.get('mem_usage_mb')}MB  "
            f"power={power_w:.1f}W",
            flush=True,
        )

        # Stop the service and sleep before next model.
        stop_services([service])
        print(
            f"    Sleeping {INTER_MODEL_SLEEP_S}s before next model...",
            flush=True,
        )
        time.sleep(INTER_MODEL_SLEEP_S)

    return results


def profile_combined_vram(
    models: list[ModelDef],
    services: list[str],
) -> dict[str, Any]:
    """Profile VRAM with all models in the wave running simultaneously.

    Args:
        models: List of ModelDef objects for the wave.
        services: List of corresponding docker-compose service names.

    Returns:
        Dict with combined VRAM and power measurements.
    """
    print("\n  --- Combined VRAM: all models simultaneous ---", flush=True)

    baseline_gpu = gpu_stats()
    baseline_vram = (
        baseline_gpu.get("vram_used_mb", 0.0) if baseline_gpu else 0.0
    )

    t_start = time.perf_counter()
    start_services(services)

    # Wait for all models to become healthy.
    health_results = {}
    for model in models:
        ok, _ = wait_for_health(model.port, timeout=START_TIMEOUT_S)
        health_results[model.name] = ok
        if not ok:
            print(
                f"    WARNING: {model.name} did not become healthy.",
                flush=True,
            )

    all_healthy_s = time.perf_counter() - t_start

    # Measure combined GPU state.
    combined_gpu = gpu_stats()
    combined_vram = (
        combined_gpu.get("vram_used_mb", 0.0) if combined_gpu else 0.0
    )
    total_vram_delta = combined_vram - baseline_vram
    power_w = (
        combined_gpu.get("power_draw_w", 0.0) if combined_gpu else 0.0
    )

    # Per-container CPU RAM.
    per_container_ram = {}
    for model in models:
        stats = docker_stats(model.container_name)
        per_container_ram[model.name] = stats.get("mem_usage_mb")

    result = {
        "n_models": len(models),
        "services": services,
        "all_healthy_time_s": round(all_healthy_s, 2),
        "health_results": health_results,
        "total_vram_delta_mb": round(total_vram_delta, 1),
        "total_vram_used_mb": round(combined_vram, 1),
        "power_draw_w": round(power_w, 1) if power_w else None,
        "per_container_cpu_ram_mb": per_container_ram,
        "gpu_stats": combined_gpu,
    }

    print(
        f"    all_healthy={all_healthy_s:.1f}s  "
        f"total_vram_delta={total_vram_delta:.1f}MB  "
        f"power={power_w:.1f}W",
        flush=True,
    )

    return result


def profile_latency(
    models: list[ModelDef],
    modality: str,
    payloads: list[tuple[str, str]],
) -> dict[str, Any]:
    """Run latency profiling against all models (assumed already running).

    For each model and input size, performs warmup runs followed by
    timed runs and records detailed latency statistics.

    Args:
        models: List of ModelDef objects (containers must be running).
        modality: One of 'image', 'audio', 'video'.
        payloads: List of (label, base64_data) tuples.

    Returns:
        Dict mapping model name -> input_size -> latency stats.
    """
    payload_key = _payload_key_for_modality(modality)
    results = {}

    for model in models:
        model_results = {}
        url = f"http://localhost:{model.port}/predict"
        print(f"\n  --- Latency: {model.name} ({url}) ---", flush=True)

        for label, b64_data in payloads:
            print(
                f"    Input size: {label}  "
                f"({WARMUP_RUNS} warmup + {TIMED_RUNS} timed)",
                flush=True,
            )

            body = {payload_key: b64_data, "threshold": 0.5}

            # Warmup runs (not timed).
            for i in range(WARMUP_RUNS):
                try:
                    requests.post(
                        url,
                        json=body,
                        timeout=REQUEST_TIMEOUT_S,
                    )
                except Exception as e:
                    print(
                        f"      Warmup {i + 1}/{WARMUP_RUNS} failed: {e}",
                        flush=True,
                    )

            # Timed runs.
            latencies = []
            errors = 0
            for i in range(TIMED_RUNS):
                t0 = time.perf_counter()
                try:
                    resp = requests.post(
                        url,
                        json=body,
                        timeout=REQUEST_TIMEOUT_S,
                    )
                    elapsed = time.perf_counter() - t0
                    if resp.status_code == 200:
                        latencies.append(elapsed)
                    else:
                        errors += 1
                        print(
                            f"      Run {i + 1}: HTTP {resp.status_code}",
                            flush=True,
                        )
                except Exception as e:
                    errors += 1
                    print(
                        f"      Run {i + 1}: error: {str(e)[:120]}",
                        flush=True,
                    )

            # Compute statistics.
            if latencies:
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)
                mean_lat = sum(sorted_lat) / n
                throughput = n / sum(sorted_lat) if sum(sorted_lat) > 0 else 0

                # Capture GPU state during inference window.
                inference_gpu = gpu_stats()

                stats = {
                    "n_success": n,
                    "n_errors": errors,
                    "mean_s": round(mean_lat, 4),
                    "median_s": round(
                        _percentile(sorted_lat, 50), 4
                    ),
                    "p50_s": round(_percentile(sorted_lat, 50), 4),
                    "p90_s": round(_percentile(sorted_lat, 90), 4),
                    "p95_s": round(_percentile(sorted_lat, 95), 4),
                    "p99_s": round(_percentile(sorted_lat, 99), 4),
                    "min_s": round(sorted_lat[0], 4),
                    "max_s": round(sorted_lat[-1], 4),
                    "stdev_s": round(_compute_stdev(sorted_lat), 4),
                    "throughput_rps": round(throughput, 3),
                    "gpu_stats_during": inference_gpu,
                }
                print(
                    f"      mean={mean_lat:.3f}s  "
                    f"p50={stats['p50_s']}s  "
                    f"p95={stats['p95_s']}s  "
                    f"p99={stats['p99_s']}s  "
                    f"rps={throughput:.2f}",
                    flush=True,
                )
            else:
                stats = {
                    "n_success": 0,
                    "n_errors": errors,
                    "error": "all runs failed",
                }
                print(f"      ALL RUNS FAILED", flush=True)

            model_results[label] = stats

        results[model.name] = model_results

    return results


# ---- Main Orchestration ----


def run_wave_profile(modality: str) -> dict[str, Any]:
    """Run the full profiling pipeline for a single modality wave.

    Steps:
        1. Build all containers for the wave.
        2. Generate synthetic test payloads.
        3. Profile individual VRAM (one model at a time).
        4. Profile combined VRAM (all models running).
        5. Run latency benchmarks (all models running).
        6. Stop all services and save results.

    Args:
        modality: One of 'image', 'audio', 'video'.

    Returns:
        Full profiling results dict.
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70, flush=True)
    print(f"  DeepSafe Wave Profiler: {modality.upper()}", flush=True)
    print(f"  Started: {timestamp}", flush=True)
    print("=" * 70, flush=True)

    models = get_models_by_modality(modality)
    service_defs = get_docker_services(modality)
    services = [s.docker_service for s in service_defs]

    print(
        f"\n  Models ({len(models)}): "
        f"{', '.join(m.name for m in models)}",
        flush=True,
    )
    print(f"  Services: {', '.join(services)}", flush=True)

    # Phase 1: Build.
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 1: Build Containers", flush=True)
    print("=" * 70, flush=True)
    t0 = time.perf_counter()
    build_results = build_services(services)
    build_time = time.perf_counter() - t0
    print(f"  Build time: {build_time:.1f}s", flush=True)
    # Filter to only successfully built services
    services = [s for s in services if build_results.get(s, False)]
    models = [m for m in models if m.docker_service in services]
    if not services:
        print("  ERROR: No services built successfully. Aborting.", flush=True)
        return

    # Phase 2: Generate payloads.
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 2: Generate Test Payloads", flush=True)
    print("=" * 70, flush=True)
    payloads = generate_payloads(modality)
    payload_summary = [
        {"label": label, "size_bytes": len(b64)}
        for label, b64 in payloads
    ]
    summary_str = ", ".join(
        f"{s['label']}={s['size_bytes'] // 1024}KB"
        for s in payload_summary
    )
    print(f"  Payload summary: {summary_str}", flush=True)

    # Phase 3: Individual VRAM profiling.
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 3: Individual VRAM Profiling", flush=True)
    print("=" * 70, flush=True)
    t0 = time.perf_counter()
    individual_vram = profile_individual_vram(models, services)
    individual_time = time.perf_counter() - t0
    print(f"  Individual profiling time: {individual_time:.1f}s", flush=True)

    # Phase 4: Combined VRAM profiling.
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 4: Combined VRAM Profiling", flush=True)
    print("=" * 70, flush=True)
    t0 = time.perf_counter()
    combined_vram = profile_combined_vram(models, services)
    combined_time = time.perf_counter() - t0
    print(f"  Combined profiling time: {combined_time:.1f}s", flush=True)

    # Phase 5: Latency profiling (models still running from Phase 4).
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 5: Latency Profiling", flush=True)
    print(
        f"  {WARMUP_RUNS} warmup + {TIMED_RUNS} timed per model per size",
        flush=True,
    )
    print("=" * 70, flush=True)
    t0 = time.perf_counter()
    latency_results = profile_latency(models, modality, payloads)
    latency_time = time.perf_counter() - t0
    print(f"  Latency profiling time: {latency_time:.1f}s", flush=True)

    # Phase 6: Stop all and save.
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 6: Cleanup & Save", flush=True)
    print("=" * 70, flush=True)
    stop_services(services)

    total_time = build_time + individual_time + combined_time + latency_time

    report = {
        "metadata": {
            "modality": modality,
            "timestamp": timestamp,
            "n_models": len(models),
            "model_names": [m.name for m in models],
            "services": services,
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
            "payload_sizes": payload_summary,
            "build_time_s": round(build_time, 1),
            "individual_vram_time_s": round(individual_time, 1),
            "combined_vram_time_s": round(combined_time, 1),
            "latency_time_s": round(latency_time, 1),
            "total_time_s": round(total_time, 1),
        },
        "individual_vram": individual_vram,
        "combined_vram": combined_vram,
        "latency": latency_results,
    }

    # Save results.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{modality}_profiling.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved: {output_path}", flush=True)

    print(f"\n  Total profiling time: {total_time:.0f}s", flush=True)
    print("  Done.", flush=True)

    return report


def main() -> None:
    """Parse CLI arguments and run the wave profiler."""
    parser = argparse.ArgumentParser(
        description="VRAM and latency profiler for DeepSafe modality waves.",
    )
    parser.add_argument(
        "--modality",
        choices=["image", "audio", "video"],
        required=True,
        help="Which modality wave to profile.",
    )
    args = parser.parse_args()
    run_wave_profile(args.modality)


if __name__ == "__main__":
    main()
