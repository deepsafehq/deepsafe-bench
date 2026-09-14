"""Monolith configuration: model registry, device config, paths.

All models reference code and weights under ``models/<modality>/<model>/{code,weights}/``.
Environment variables override defaults for deployment flexibility.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS_ROOT = Path(os.getenv("DEEPSAFE_MODELS_ROOT", str(REPO_ROOT / "models")))

GATEWAY_ROOT = Path(
    os.getenv("DEEPSAFE_GATEWAY_ROOT", str(REPO_ROOT / "apps" / "gateway"))
)


# ── Device ───────────────────────────────────────────────────────────────────

DEVICE = os.getenv("DEEPSAFE_DEVICE", "auto")  # auto | cuda | mps | cpu

# Models to load: "all" or comma-separated names like "npr,aide,sbi"
ENABLED_MODELS = os.getenv("DEEPSAFE_MODELS", "all")

# Server
PORT = int(os.getenv("DEEPSAFE_PORT", "8000"))
HOST = os.getenv("DEEPSAFE_HOST", "0.0.0.0")

# Inference
MAX_WORKERS = int(os.getenv("DEEPSAFE_MAX_WORKERS", "3"))


# ── Model definitions ────────────────────────────────────────────────────────


@dataclass
class ModelDef:
    """Metadata for a single detection model."""

    name: str
    modality: str  # image | video | audio | provenance
    model_dir: str  # relative to MODELS_ROOT, e.g. "image/npr"
    needs_isolation: bool = False
    gpu: bool = True
    extra: Dict = field(default_factory=dict)


# ── Model Registry ───────────────────────────────────────────────────────────
# Every model the monolith can load. Order determines load order.

MODEL_REGISTRY: Dict[str, ModelDef] = {
    # ── Image (7) ────────────────────────────────────────────────────────
    "npr": ModelDef(
        name="npr",
        modality="image",
        model_dir="image/npr",
    ),
    "yermandy": ModelDef(
        name="yermandy",
        modality="image",
        model_dir="image/yermandy",
    ),
    "universal": ModelDef(
        name="universal",
        modality="image",
        model_dir="image/universal",
    ),
    "aide": ModelDef(
        name="aide",
        modality="image",
        model_dir="image/aide",
        needs_isolation=True,
        extra={"checkpoint": "GenImage_train.pth"},
    ),
    "fsd": ModelDef(
        name="fsd",
        modality="image",
        model_dir="image/fsd",
    ),
    "effort": ModelDef(
        name="effort",
        modality="image",
        model_dir="image/effort",
    ),
    "cospy": ModelDef(
        name="cospy",
        modality="image",
        model_dir="image/cospy",
    ),
    # ── Video (7) ────────────────────────────────────────────────────────
    "fakestormer": ModelDef(
        name="fakestormer",
        modality="video",
        model_dir="video/fakestormer",
        needs_isolation=True,
    ),
    "sbi": ModelDef(
        name="sbi",
        modality="video",
        model_dir="video/sbi",
    ),
    "dfd_fcg": ModelDef(
        name="dfd_fcg",
        modality="video",
        model_dir="video/dfd_fcg",
    ),
    "pwtf_dvd": ModelDef(
        name="pwtf_dvd",
        modality="video",
        model_dir="video/pwtf_dvd",
    ),
    "lipfd": ModelDef(
        name="lipfd",
        modality="video",
        model_dir="video/lipfd",
        needs_isolation=True,
    ),
    "recce": ModelDef(
        name="recce",
        modality="video",
        model_dir="video/recce",
        needs_isolation=True,
    ),
    "mintime": ModelDef(
        name="mintime",
        modality="video",
        model_dir="video/mintime",
        needs_isolation=True,
    ),
    # ── Video: Full-Frame Detectors (no face detection needed) ──────────
    # These complement the face-dependent models above by detecting
    # AI-generated video content (Sora, Kling, Veo) without faces.
    "npr_video": ModelDef(
        name="npr_video",
        modality="video",
        model_dir="video/npr_video",  # code/ and weights/ symlink to image/npr
    ),
    "univfd_video": ModelDef(
        name="univfd_video",
        modality="video",
        model_dir="video/univfd_video",  # code/ and weights/ symlink to image/universal
    ),
    # ── Audio (3) ────────────────────────────────────────────────────────
    # SONICS removed: designed for AI-generated music (Suno/Udio), not speech deepfakes.
    # AASIST3 removed: KAN/GAT model construction hangs indefinitely (size=200 param).
    #   Individual AUC=0.199 (anti-correlated). RF used it as contrarian (+0.015 AUC)
    #   but not worth the deployment risk. Can re-add if initialization is fixed.
    "shiftyspeech": ModelDef(
        name="shiftyspeech",
        modality="audio",
        model_dir="audio/shiftyspeech",
    ),
    "safeear": ModelDef(
        name="safeear",
        modality="audio",
        model_dir="audio/safeear",
    ),
    "nes2net": ModelDef(
        name="nes2net",
        modality="audio",
        model_dir="audio/nes2net",
    ),
    # ── Provenance (5) ───────────────────────────────────────────────────
    "c2pa": ModelDef(
        name="c2pa",
        modality="provenance",
        model_dir="provenance/c2pa",
        gpu=False,
    ),
    "sdxl_watermark": ModelDef(
        name="sdxl_watermark",
        modality="provenance",
        model_dir="provenance/sdxl_watermark",
        gpu=False,
    ),
    "audioseal": ModelDef(
        name="audioseal",
        modality="provenance",
        model_dir="provenance/audioseal",
    ),
    "videoseal": ModelDef(
        name="videoseal",
        modality="provenance",
        model_dir="provenance/videoseal",
    ),
    "trustmark": ModelDef(
        name="trustmark",
        modality="provenance",
        model_dir="provenance/trustmark",
        gpu=False,
    ),
}


def get_enabled_models() -> List[str]:
    """Return list of model names to load based on ENABLED_MODELS env var."""
    if ENABLED_MODELS == "all":
        return list(MODEL_REGISTRY.keys())
    return [m.strip() for m in ENABLED_MODELS.split(",") if m.strip()]


def get_models_by_modality(modality: str) -> List[str]:
    """Return model names for a given modality."""
    return [name for name, defn in MODEL_REGISTRY.items() if defn.modality == modality]


def get_model_dir(model_name: str) -> Path:
    """Return absolute path to a model's root directory."""
    defn = MODEL_REGISTRY[model_name]
    return MODELS_ROOT / defn.model_dir


def get_model_code_path(model_name: str) -> Path:
    """Return absolute path to a model's code directory."""
    defn = MODEL_REGISTRY[model_name]
    return MODELS_ROOT / defn.model_dir / "code"


def get_weights_path(model_name: str) -> Path:
    """Return absolute path to a model's weights directory."""
    defn = MODEL_REGISTRY[model_name]
    return MODELS_ROOT / defn.model_dir / "weights"
