"""Base predictor interface and device utilities.

Every model wrapper inherits from BasePredictor and implements
load(), predict(), and health().
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from deepsafe_shared.device import get_device, setup_inference_optimizations

logger = logging.getLogger("models")


# ── Base Predictor ───────────────────────────────────────────────────────────


class BasePredictor(ABC):
    """Abstract base class for all model wrappers."""

    name: str = ""
    modality: str = ""
    _loaded: bool = False
    _device: Optional[torch.device] = None
    _load_time_ms: float = 0.0
    _use_amp: bool = True  # automatic mixed precision (FP16 on CUDA)

    @abstractmethod
    def load(self, weights_dir: Path, device: torch.device) -> None:
        """Load model weights onto the given device."""

    @abstractmethod
    def predict(self, raw_bytes: bytes) -> dict:
        """Run inference on raw file bytes."""

    def predict_preprocessed(self, video_data) -> dict:
        """Run inference using shared preprocessed video data.

        Override in video models to skip redundant decoding and
        face detection.  Default falls back to predict(raw_bytes).
        """
        return self.predict(video_data.raw_bytes)

    def health(self) -> dict:
        info = {
            "name": self.name,
            "modality": self.modality,
            "loaded": self._loaded,
            "device": str(self._device) if self._device else "none",
            "load_time_ms": self._load_time_ms,
        }
        if self._device and self._device.type == "cuda":
            info.update(
                {
                    "vram_used_mb": round(torch.cuda.memory_allocated(0) / 1024**2),
                    "vram_total_mb": round(
                        torch.cuda.get_device_properties(0).total_memory / 1024**2
                    ),
                }
            )
        return info

    def _optimize_for_inference(
        self,
        model: torch.nn.Module,
        use_compile: bool = True,
        use_channels_last: bool = False,
    ) -> torch.nn.Module:
        """Apply inference speed optimizations to a model.

        On CUDA: converts weights to FP16 (BatchNorm stays FP32),
        optionally enables channels_last memory format for 2D CNNs,
        and applies torch.compile for kernel fusion.

        Args:
            model: The PyTorch model to optimize.
            use_compile: Apply torch.compile (default True).
            use_channels_last: Use channels_last memory (best for 2D CNNs).

        Returns:
            Optimized model.
        """
        if not self._device or self._device.type != "cuda":
            return model

        # FP16 weights: halves memory bandwidth, ~2x throughput.
        # Keep BatchNorm in FP32 for numerical stability.
        model = model.half()
        for m in model.modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                ),
            ):
                m.float()
        logger.info("%s: converted to FP16", self.name)

        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
            logger.info("%s: channels_last memory format", self.name)

        if use_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, dynamic=True)
                logger.info("%s: torch.compile enabled", self.name)
            except Exception as e:
                logger.warning(
                    "%s: torch.compile failed: %s",
                    self.name,
                    e,
                )

        return model

    def _timed_predict(self, fn, raw_bytes: bytes) -> dict:
        """Wrapper that times a predict function, applies AMP, and catches errors.

        AMP (Automatic Mixed Precision) is used on CUDA when _use_amp=True.
        If AMP fails, it falls back to FP32 automatically and disables AMP
        for future calls on this model instance.
        """
        start = time.perf_counter()
        try:
            # Use automatic mixed precision on CUDA for ~2x speedup
            if self._use_amp and self._device and self._device.type == "cuda":
                try:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        result = fn(raw_bytes)
                except Exception:
                    # AMP failed — fall back to FP32 and disable for this model
                    logger.warning("%s: AMP failed, falling back to FP32", self.name)
                    self._use_amp = False
                    result = fn(raw_bytes)
            else:
                result = fn(raw_bytes)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result["latency_ms"] = round(elapsed_ms, 1)
            return result
        except torch.cuda.CudaError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("%s CUDA error: %s", self.name, e)
            torch.cuda.empty_cache()
            return {
                "probability": None,
                "error": f"CUDA error: {e}",
                "latency_ms": round(elapsed_ms, 1),
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("%s predict error: %s", self.name, e, exc_info=True)
            return {
                "probability": None,
                "error": str(e),
                "latency_ms": round(elapsed_ms, 1),
            }
