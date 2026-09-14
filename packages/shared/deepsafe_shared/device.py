"""Universal device selection and inference optimizations for PyTorch.

Merges the gateway's MPS-first auto-detection with the monolith's
CUDA-first detection and GPU optimization setup.

Usage:
    from deepsafe_shared.device import get_device, setup_inference_optimizations

    device = get_device()                      # auto: MPS > CUDA > CPU
    device = get_device(preference="cuda_first")  # auto: CUDA > MPS > CPU
    setup_inference_optimizations(device)       # global torch settings
"""

import logging
import os
import platform
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    """Check if Apple Metal Performance Shaders backend is available."""
    return (
        platform.system() == "Darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def get_device(
    force: Optional[str] = None,
    preference: str = "auto",
) -> torch.device:
    """Select the optimal compute device for PyTorch inference.

    Args:
        force: Override device selection. One of 'cpu', 'cuda', 'mps'.
            If None, checks the DEEPSAFE_DEVICE env var, then auto-detects.
        preference: Auto-detection priority when no override is set.
            'auto' — MPS > CUDA > CPU (dev machines).
            'cuda_first' — CUDA > MPS > CPU (GPU servers).

    Returns:
        torch.device for the selected backend.
    """
    override = force or os.environ.get("DEEPSAFE_DEVICE", "").strip().lower()
    if override and override != "auto":
        if override == "cpu":
            logger.info("Device forced to CPU via override.")
            return torch.device("cpu")
        if override == "cuda":
            if torch.cuda.is_available():
                logger.info("Device forced to CUDA via override.")
                return torch.device("cuda")
            logger.warning("CUDA forced but not available. Falling back to CPU.")
            return torch.device("cpu")
        if override == "mps":
            if _mps_available():
                logger.info("Device forced to MPS via override.")
                return torch.device("mps")
            logger.warning("MPS forced but not available. Falling back to CPU.")
            return torch.device("cpu")

    # Auto-detect based on preference
    if preference == "cuda_first":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("Auto-detected NVIDIA GPU: %s. Using CUDA.", gpu_name)
            return torch.device("cuda")
        if _mps_available():
            logger.info("Auto-detected Apple Silicon GPU (MPS).")
            return torch.device("mps")
    else:
        # Default: MPS > CUDA > CPU (for dev machines)
        if _mps_available():
            logger.info(
                "Auto-detected Apple Silicon GPU (MPS). "
                "Using Metal Performance Shaders."
            )
            return torch.device("mps")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("Auto-detected NVIDIA GPU: %s. Using CUDA.", gpu_name)
            return torch.device("cuda")

    logger.info("No GPU detected. Using CPU.")
    return torch.device("cpu")


def get_map_location(device: Optional[torch.device] = None) -> str:
    """Get the appropriate map_location for torch.load().

    Args:
        device: Target device. If None, auto-detects via get_device().

    Returns:
        String suitable for torch.load(map_location=...).
    """
    if device is None:
        device = get_device()
    return str(device)


def setup_inference_optimizations(device: torch.device) -> None:
    """Apply global inference optimizations at startup.

    Disables gradient computation, enables cuDNN benchmark mode,
    TF32 precision on Ampere+ GPUs, and FP16 reduction.

    Args:
        device: The active compute device.
    """
    torch.set_grad_enabled(False)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduction"):
                torch.backends.cuda.matmul.allow_fp16_reduction = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)
    elif device.type == "mps":
        logger.info("Device: Apple MPS")
    else:
        logger.info("Device: CPU")
