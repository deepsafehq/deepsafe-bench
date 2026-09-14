"""GPU monitoring and container lifecycle utilities for benchmarking.

Provides functions to query nvidia-smi, collect docker stats, wait
for container health, and measure per-container VRAM consumption.
Designed for the RTX PRO 6000 (96 GB) benchmarking VM.
"""

import logging
import subprocess
import time
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NVIDIA_SMI_QUERY = (
    "name,memory.used,memory.total,"
    "utilization.gpu,power.draw,temperature.gpu"
)
_NVIDIA_SMI_CMD = [
    "nvidia-smi",
    f"--query-gpu={_NVIDIA_SMI_QUERY}",
    "--format=csv,noheader,nounits",
]

_DEFAULT_HEALTH_TIMEOUT = 300   # seconds
_DEFAULT_HEALTH_INTERVAL = 5    # seconds
_SUBPROCESS_TIMEOUT = 30        # seconds


# ---------------------------------------------------------------------------
# GPU stats
# ---------------------------------------------------------------------------

def gpu_stats() -> Dict[str, object]:
    """Query nvidia-smi for current GPU telemetry.

    Returns:
        Dict with keys: gpu_name, vram_used_mb, vram_total_mb,
        gpu_util_pct, power_draw_w, temperature_c.
        Returns an empty dict if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            _NVIDIA_SMI_CMD,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except FileNotFoundError:
        logger.warning("nvidia-smi not found; GPU stats unavailable.")
        return {}
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out after %ds.", _SUBPROCESS_TIMEOUT)
        return {}

    if result.returncode != 0:
        logger.warning(
            "nvidia-smi exited with code %d: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return {}

    line = result.stdout.strip()
    if not line:
        logger.warning("nvidia-smi returned empty output.")
        return {}

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        logger.warning(
            "nvidia-smi returned unexpected format: %s", line
        )
        return {}

    return {
        "gpu_name": parts[0],
        "vram_used_mb": float(parts[1]),
        "vram_total_mb": float(parts[2]),
        "gpu_util_pct": float(parts[3]),
        "power_draw_w": float(parts[4]),
        "temperature_c": float(parts[5]),
    }


# ---------------------------------------------------------------------------
# Docker stats
# ---------------------------------------------------------------------------

def docker_stats(container_name: str) -> Dict[str, object]:
    """Collect resource usage for a running Docker container.

    Args:
        container_name: Full container name (e.g. "deepsafe-npr_deepfakedetection").

    Returns:
        Dict with keys: memory_usage (str, e.g. "1.23GiB"), cpu_pct (float).
        Returns an empty dict on failure.
    """
    cmd = [
        "docker", "stats", "--no-stream",
        "--format", "{{.MemUsage}}\t{{.CPUPerc}}",
        container_name,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except FileNotFoundError:
        logger.warning("docker CLI not found.")
        return {}
    except subprocess.TimeoutExpired:
        logger.warning(
            "docker stats timed out for %s.", container_name
        )
        return {}

    if result.returncode != 0:
        logger.warning(
            "docker stats failed for %s: %s",
            container_name,
            result.stderr.strip(),
        )
        return {}

    line = result.stdout.strip()
    if not line:
        return {}

    parts = line.split("\t")
    if len(parts) < 2:
        logger.warning(
            "docker stats unexpected format for %s: %s",
            container_name,
            line,
        )
        return {}

    # Memory format: "1.23GiB / 4GiB" -- take the used portion.
    mem_raw = parts[0].strip()
    memory_usage = mem_raw.split("/")[0].strip() if "/" in mem_raw else mem_raw

    # CPU format: "12.34%" -- strip the percent sign.
    cpu_raw = parts[1].strip().rstrip("%")
    try:
        cpu_pct = float(cpu_raw)
    except ValueError:
        cpu_pct = 0.0

    return {
        "memory_usage": memory_usage,
        "cpu_pct": cpu_pct,
    }


# ---------------------------------------------------------------------------
# Health polling
# ---------------------------------------------------------------------------

def wait_for_health(
    port: int,
    timeout: int = _DEFAULT_HEALTH_TIMEOUT,
    interval: int = _DEFAULT_HEALTH_INTERVAL,
) -> Tuple[bool, float]:
    """Poll a service health endpoint until healthy or timeout.

    Args:
        port: Localhost port to poll (e.g. 5001).
        timeout: Maximum seconds to wait.
        interval: Seconds between polls.

    Returns:
        Tuple of (healthy: bool, seconds_waited: float).
    """
    url = f"http://localhost:{port}/health"
    start = time.monotonic()
    deadline = start + timeout

    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                elapsed = time.monotonic() - start
                logger.info(
                    "Port %d healthy after %.1fs.", port, elapsed
                )
                return True, elapsed
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            pass

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval, remaining))

    elapsed = time.monotonic() - start
    logger.warning("Port %d not healthy after %.1fs.", port, elapsed)
    return False, elapsed


# ---------------------------------------------------------------------------
# Full container VRAM measurement
# ---------------------------------------------------------------------------

def _get_docker_image_size(container_name: str) -> Optional[str]:
    """Return the compressed image size for a container.

    Args:
        container_name: Running container name.

    Returns:
        Human-readable size string (e.g. "2.14GB"), or None.
    """
    cmd = [
        "docker", "inspect",
        "--format", "{{.Image}}",
        container_name,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    image_id = result.stdout.strip()
    if not image_id:
        return None

    size_cmd = [
        "docker", "image", "inspect",
        "--format", "{{.Size}}",
        image_id,
    ]
    try:
        size_result = subprocess.run(
            size_cmd, capture_output=True, text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if size_result.returncode != 0:
        return None

    raw = size_result.stdout.strip()
    try:
        size_bytes = int(raw)
        return f"{size_bytes / 1e9:.2f}GB"
    except ValueError:
        return raw


def measure_vram_for_container(
    container_name: str,
    port: int,
    compose_file: str,
    prod_file: str,
    service_name: str,
) -> Dict[str, object]:
    """Measure VRAM footprint of a single model container.

    Records baseline GPU VRAM, starts the container via docker compose,
    waits for the health endpoint, then records loaded VRAM, docker
    resource stats, and image size.

    Args:
        container_name: Docker container name.
        port: Localhost port the service listens on.
        compose_file: Path to base docker-compose.yml.
        prod_file: Path to docker-compose.prod.yml with GPU overrides.
        service_name: Docker Compose service name to start.

    Returns:
        Dict with keys: service_name, container_name, port,
        baseline_vram_mb, loaded_vram_mb, vram_delta_mb,
        healthy, startup_seconds, docker_memory, docker_cpu_pct,
        image_size, gpu_snapshot.
    """
    # 1. Baseline GPU state.
    baseline = gpu_stats()
    baseline_vram = baseline.get("vram_used_mb", 0.0)

    # 2. Start the container.
    compose_cmd = [
        "docker", "compose",
        "-f", compose_file,
        "-f", prod_file,
        "up", "-d", service_name,
    ]
    logger.info("Starting %s via docker compose...", service_name)
    try:
        result = subprocess.run(
            compose_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(
                "docker compose up failed for %s: %s",
                service_name,
                result.stderr.strip(),
            )
            return {
                "service_name": service_name,
                "container_name": container_name,
                "port": port,
                "error": result.stderr.strip(),
            }
    except subprocess.TimeoutExpired:
        logger.error(
            "docker compose up timed out for %s.", service_name
        )
        return {
            "service_name": service_name,
            "container_name": container_name,
            "port": port,
            "error": "docker compose up timed out",
        }

    # 3. Wait for health.
    healthy, startup_secs = wait_for_health(port)

    # 4. Loaded GPU state.
    loaded = gpu_stats()
    loaded_vram = loaded.get("vram_used_mb", 0.0)

    # 5. Docker resource stats.
    dstats = docker_stats(container_name)

    # 6. Image size.
    image_size = _get_docker_image_size(container_name)

    return {
        "service_name": service_name,
        "container_name": container_name,
        "port": port,
        "baseline_vram_mb": baseline_vram,
        "loaded_vram_mb": loaded_vram,
        "vram_delta_mb": loaded_vram - baseline_vram,
        "healthy": healthy,
        "startup_seconds": round(startup_secs, 2),
        "docker_memory": dstats.get("memory_usage", "N/A"),
        "docker_cpu_pct": dstats.get("cpu_pct", 0.0),
        "image_size": image_size or "N/A",
        "gpu_snapshot": loaded,
    }
