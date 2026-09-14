"""DeepSafe Monolith Inference Server.

Single FastAPI process serving all 22 detection models.
Models are called via direct Python function calls (no HTTP/base64).

Endpoints:
    GET  /health             - Server + all model health
    GET  /models             - List loaded models
    POST /models/{name}/predict - Single model inference (base64 JSON)
    POST /v1/detect          - Full ensemble pipeline (multipart file)

Usage:
    python -m monolith.server                    # start server
    DEEPSAFE_MODELS=npr,aide python -m monolith.server  # subset
"""

import base64
import logging
import os
import sys
import time

# Ensure this package and shared are on sys.path regardless of how
# the server is started (direct python, uvicorn, systemd, etc.).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in [_THIS_DIR, os.path.join(_REPO_ROOT, "packages", "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import torch
import uvicorn
from config import (
    HOST,
    MAX_WORKERS,
    MODEL_REGISTRY,
    PORT,
    get_enabled_models,
    get_models_by_modality,
    get_weights_path,
)
from deepsafe_shared.device import get_device, setup_inference_optimizations
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from models.base import BasePredictor
from pydantic import BaseModel, Field

from models import get_predictor_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("server")

# ── Global state ─────────────────────────────────────────────────────────────

_models: Dict[str, BasePredictor] = {}
_failed_models: list[str] = []
_device: Optional[torch.device] = None
_executor: Optional[ThreadPoolExecutor] = None

# ── Pydantic models ──────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Request for single-model prediction (backward compat with Docker)."""

    image_data: Optional[str] = Field(None, description="Base64 image")
    audio_data: Optional[str] = Field(None, description="Base64 audio")
    video_data: Optional[str] = Field(None, description="Base64 video")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="DeepSafe Monolith Inference Server",
    description="Consolidated inference for all 22 DeepSafe detection models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup / Shutdown ───────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    """Load all enabled models into GPU/CPU memory."""
    global _device, _executor

    _device = get_device()
    setup_inference_optimizations(_device)

    logger.info("=" * 60)
    logger.info("DeepSafe Monolith — Loading models on %s", _device)
    logger.info("=" * 60)

    model_names = get_enabled_models()
    logger.info("Models to load: %s", model_names)

    _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    for name in model_names:
        try:
            cls = get_predictor_class(name)
            predictor = cls()
            weights_dir = get_weights_path(name)
            start = time.perf_counter()
            predictor.load(weights_dir, _device)
            elapsed = (time.perf_counter() - start) * 1000
            predictor._load_time_ms = elapsed
            _models[name] = predictor
            logger.info("  %s: LOADED (%.0fms)", name, elapsed)
        except Exception as e:
            logger.error("  %s: FAILED — %s", name, e, exc_info=True)
            _failed_models.append(name)

    logger.info("-" * 60)
    logger.info(
        "Startup complete: %d/%d models loaded, %d failed",
        len(_models),
        len(model_names),
        len(_failed_models),
    )
    if _failed_models:
        logger.warning("Failed models: %s", _failed_models)

    # Report VRAM usage
    if _device and _device.type == "cuda":
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("VRAM: %.1f / %.1f GB", vram_used, vram_total)


@app.on_event("shutdown")
async def shutdown():
    """Clean up resources."""
    if _executor:
        _executor.shutdown(wait=False)
    logger.info("Server shutting down.")


# ── Health endpoint ──────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Server and model health status."""
    model_health = {name: model.health() for name, model in _models.items()}
    gpu_info = {}
    if _device and _device.type == "cuda":
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_used_mb": round(torch.cuda.memory_allocated(0) / 1024**2),
            "vram_total_mb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2
            ),
        }
    return {
        "status": "healthy" if _models else "degraded",
        "device": str(_device),
        "models_loaded": len(_models),
        "models_failed": _failed_models,
        "models": model_health,
        **gpu_info,
    }


@app.get("/models")
async def list_models():
    """List all loaded models with their modality."""
    return {
        name: {
            "modality": model.modality,
            "loaded": model._loaded,
            "device": str(model._device),
        }
        for name, model in _models.items()
    }


# ── Single model predict ────────────────────────────────────────────────────


@app.post("/models/{model_name}/predict")
async def predict_single(model_name: str, request: PredictRequest):
    """Run inference on a single model.

    Backward compatible with Docker service /predict endpoints.
    Accepts base64-encoded media in image_data, audio_data, or video_data.
    """
    if model_name not in _models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not loaded. "
            f"Available: {list(_models.keys())}",
        )

    # Extract raw bytes from whichever field is populated
    raw_bytes = None
    for field_name in ("image_data", "audio_data", "video_data"):
        data = getattr(request, field_name, None)
        if data:
            raw_bytes = base64.b64decode(data)
            break

    if raw_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="No media data provided. Set image_data, audio_data, "
            "or video_data with base64-encoded content.",
        )

    result = _models[model_name].predict(raw_bytes)
    result["model"] = model_name
    return result


# ── Full ensemble detect ─────────────────────────────────────────────────────

_MIME_TO_MODALITY = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
    "image/gif": "image",
    "image/tiff": "image",
    "image/bmp": "image",
    "image/heic": "image",
    "image/heif": "image",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/flac": "audio",
    "audio/ogg": "audio",
    "audio/mp4": "audio",
    "audio/aac": "audio",
    "video/mp4": "video",
    "video/mpeg": "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
    "video/webm": "video",
    "video/x-matroska": "video",
}


def _detect_media_type(filename: str, content_type: str) -> str:
    """Determine media type from filename and content type."""
    # Try content type first
    if content_type and content_type in _MIME_TO_MODALITY:
        return _MIME_TO_MODALITY[content_type]

    # Fall back to extension
    ext = Path(filename).suffix.lower() if filename else ""
    ext_map = {
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".webp": "image",
        ".gif": "image",
        ".tiff": "image",
        ".bmp": "image",
        ".heic": "image",
        ".heif": "image",
        ".wav": "audio",
        ".mp3": "audio",
        ".flac": "audio",
        ".ogg": "audio",
        ".aac": "audio",
        ".m4a": "audio",
        ".mp4": "video",
        ".avi": "video",
        ".mov": "video",
        ".mkv": "video",
        ".webm": "video",
    }
    return ext_map.get(ext, "image")  # Default to image


@app.post("/v1/detect")
async def detect(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    """Full ensemble detection pipeline.

    Upload a file, get back per-model results + ensemble verdict.
    Compatible with the gateway's /v1/detect endpoint.
    """
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    # Read file
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    media_type = _detect_media_type(file.filename, file.content_type)
    logger.info(
        "Request %s: %s (%s, %d bytes)",
        request_id,
        file.filename,
        media_type,
        len(raw_bytes),
    )

    # Get detection + provenance models for this media type
    detection_models = get_models_by_modality(media_type)
    provenance_models = get_models_by_modality("provenance")
    all_model_names = detection_models + provenance_models

    # Filter to loaded models only
    active_models = [n for n in all_model_names if n in _models]

    if not active_models:
        raise HTTPException(
            status_code=503,
            detail=f"No models loaded for media type '{media_type}'",
        )

    # For video: shared preprocessing (decode once, detect faces once)
    video_data = None
    if media_type == "video":
        try:
            from video_preprocess import (
                cleanup_video_data,
                preprocess_video,
            )

            video_data = preprocess_video(raw_bytes, _device)
            logger.info(
                "Request %s: shared preprocess — %d frames, %d sampled",
                request_id,
                len(video_data.frames_all),
                len(video_data.frames_8),
            )
        except Exception as e:
            logger.warning(
                "Request %s: shared preprocess failed, "
                "falling back to per-model: %s",
                request_id,
                e,
            )
            video_data = None

    # ── Pipeline execution for video: overlap CPU preprocessing with
    # GPU inference by running light models while dense models work. ──
    _DENSE_MODELS = {"dfd_fcg", "pwtf_dvd", "mintime"}

    model_results = {}
    futures = {}

    if video_data is not None and media_type == "video":
        # Start dense models in background (CPU-bound preprocessing)
        for name in active_models:
            if name in _DENSE_MODELS and name in _models:
                model = _models[name]
                future = _executor.submit(
                    model.predict_preprocessed,
                    video_data,
                )
                futures[future] = name

        # Run light models sequentially on GPU (fast, no contention)
        for name in active_models:
            if name not in _DENSE_MODELS and name in _models:
                model = _models[name]
                try:
                    if hasattr(model, "predict_preprocessed"):
                        model_results[name] = model.predict_preprocessed(video_data)
                    else:
                        model_results[name] = model.predict(raw_bytes)
                except Exception as e:
                    logger.error(
                        "Request %s: %s error: %s",
                        request_id,
                        name,
                        e,
                    )
                    model_results[name] = {
                        "probability": None,
                        "error": str(e),
                        "latency_ms": 0,
                    }

        # Collect dense model results
        for future in as_completed(futures, timeout=600):
            name = futures[future]
            try:
                model_results[name] = future.result()
            except Exception as e:
                logger.error(
                    "Request %s: %s error: %s",
                    request_id,
                    name,
                    e,
                )
                model_results[name] = {
                    "probability": None,
                    "error": str(e),
                    "latency_ms": 0,
                }
    else:
        # Non-video: fan out all models in parallel
        for name in active_models:
            future = _executor.submit(
                _models[name].predict,
                raw_bytes,
            )
            futures[future] = name

        for future in as_completed(futures, timeout=600):
            name = futures[future]
            try:
                model_results[name] = future.result()
            except Exception as e:
                logger.error(
                    "Request %s: %s error: %s",
                    request_id,
                    name,
                    e,
                )
                model_results[name] = {
                    "probability": None,
                    "error": str(e),
                    "latency_ms": 0,
                }

    # Cleanup shared video data
    if video_data is not None:
        try:
            from video_preprocess import cleanup_video_data

            cleanup_video_data(video_data)
        except Exception:
            pass

    # Compute ensemble
    from ensemble import compute_ensemble

    ensemble_result = compute_ensemble(
        model_results,
        media_type,
        threshold,
        request_id,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "id": request_id,
        "request_id": request_id,
        "filename": file.filename,
        "media_type": media_type,
        "verdict": ensemble_result["verdict"],
        "confidence": ensemble_result["confidence"],
        "score": ensemble_result["score"],
        "method": ensemble_result["method"],
        "threshold": threshold,
        "models_used": ensemble_result["models_used"],
        "model_results": {
            name: {
                "probability": r.get("probability"),
                "prediction": r.get("prediction"),
                "latency_ms": r.get("latency_ms"),
                "error": r.get("error"),
            }
            for name, r in model_results.items()
        },
        "total_latency_ms": round(elapsed_ms, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    """Start the monolith server."""
    logger.info("Starting DeepSafe Monolith on %s:%d", HOST, PORT)
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
