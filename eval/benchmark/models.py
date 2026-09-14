"""Central model registry for GPU benchmarking.

Defines all 25 containerized detection services with their metadata,
ports, CUDA tiers, and memory limits. Used by benchmark scripts to
discover, start, and measure individual model containers.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ModelDef:
    """Definition of a single detection model service."""

    name: str
    modality: str  # "image", "video", "audio", or "provenance"
    port: int
    docker_service: str
    container_name: str
    architecture: str
    paper: str
    cuda_tier: int  # 0=CPU, 1=CUDA11.1, 2=CUDA11.3, 3=CUDA12.1
    mem_limit: str
    payload_key: str  # "image_data", "video_data", or "audio_data"


# ---------------------------------------------------------------------------
# All 25 model services
# ---------------------------------------------------------------------------

ALL_MODELS: List[ModelDef] = [
    # ── Image (7) ─────────────────────────────────────────────────────────
    ModelDef(
        name="npr",
        modality="image",
        port=5001,
        docker_service="npr_deepfakedetection",
        container_name="deepsafe-npr_deepfakedetection",
        architecture="NPR pixel-level noise pattern residual",
        paper="CVPR 2024",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="image_data",
    ),
    ModelDef(
        name="yermandy",
        modality="image",
        port=5002,
        docker_service="yermandy_clip_detection",
        container_name="deepsafe-yermandy_clip_detection",
        architecture="CLIP ViT-L/14 linear probe",
        paper="ICLR 2024 Workshop",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="image_data",
    ),
    ModelDef(
        name="universal",
        modality="image",
        port=5003,
        docker_service="universalfakedetect",
        container_name="deepsafe-universalfakedetect",
        architecture="OpenAI CLIP ViT-L/14 nearest-neighbor",
        paper="CVPR 2023",
        cuda_tier=2,
        mem_limit="4g",
        payload_key="image_data",
    ),
    ModelDef(
        name="aide",
        modality="image",
        port=5004,
        docker_service="aide_detection",
        container_name="deepsafe-aide_detection",
        architecture="AIDE blended image + frequency features",
        paper="CVPR 2024",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="image_data",
    ),
    ModelDef(
        name="fsd",
        modality="image",
        port=5005,
        docker_service="fsd_detection",
        container_name="deepsafe-fsd_detection",
        architecture="FSD frequency-space decomposition",
        paper="AAAI 2024",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="image_data",
    ),
    ModelDef(
        name="effort",
        modality="image",
        port=5006,
        docker_service="effort_detection",
        container_name="deepsafe-effort-detection",
        architecture="Effort SVD + CLIP feature alignment",
        paper="ICML 2025",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="image_data",
    ),
    ModelDef(
        name="cospy",
        modality="image",
        port=5007,
        docker_service="cospy_detection",
        container_name="deepsafe-cospy-detection",
        architecture="CO-SPY SigLIP + Stable Diffusion VAE",
        paper="CVPR 2025",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="image_data",
    ),

    # ── Video (7) ─────────────────────────────────────────────────────────
    ModelDef(
        name="fakestormer",
        modality="video",
        port=7001,
        docker_service="fakestormer_detection",
        container_name="deepsafe-fakestormer-detection",
        architecture="FakeSTormer spatial-temporal transformer + mmcv",
        paper="CVPR 2024",
        cuda_tier=1,
        mem_limit="6g",
        payload_key="video_data",
    ),
    ModelDef(
        name="sbi",
        modality="video",
        port=7002,
        docker_service="sbi_detection",
        container_name="deepsafe-sbi-detection",
        architecture="SBI EfficientNet-B4 self-blended images",
        paper="CVPR 2022",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="video_data",
    ),
    ModelDef(
        name="dfd_fcg",
        modality="video",
        port=7003,
        docker_service="dfd_fcg_detection",
        container_name="deepsafe-dfd-fcg-detection",
        architecture="DFD-FCG CLIP ViT-L/14 + fine-grained classification",
        paper="CVPR 2025",
        cuda_tier=3,
        mem_limit="8g",
        payload_key="video_data",
    ),
    ModelDef(
        name="pwtf_dvd",
        modality="video",
        port=7005,
        docker_service="pwtf_dvd_detection",
        container_name="deepsafe-pwtf-dvd-detection",
        architecture="PwTF-DVD per-pixel temporal FFT + SlowFast",
        paper="ICCV 2025",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="video_data",
    ),
    ModelDef(
        name="lipfd",
        modality="video",
        port=7006,
        docker_service="lipfd_detection",
        container_name="deepsafe-lipfd-detection",
        architecture="LipFD CLIP ViT-L/14 multi-scale lip region",
        paper="NeurIPS 2024",
        cuda_tier=3,
        mem_limit="8g",
        payload_key="video_data",
    ),
    ModelDef(
        name="recce",
        modality="video",
        port=7007,
        docker_service="recce_detection",
        container_name="deepsafe-recce-detection",
        architecture="RECCE reconstruction-based anomaly detection",
        paper="CVPR 2022",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="video_data",
    ),
    ModelDef(
        name="mintime",
        modality="video",
        port=7008,
        docker_service="mintime_detection",
        container_name="deepsafe-mintime-detection",
        architecture="MINTIME TimeSformer multi-identity temporal",
        paper="IEEE T-IFS 2024",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="video_data",
    ),

    # ── Audio (5) ─────────────────────────────────────────────────────────
    ModelDef(
        name="shiftyspeech",
        modality="audio",
        port=8001,
        docker_service="shiftyspeech_detection",
        container_name="deepsafe-shiftyspeech-detection",
        architecture="ShiftySpeech spectral shift detection",
        paper="AAAI 2025",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="audio_data",
    ),
    ModelDef(
        name="safeear",
        modality="audio",
        port=8002,
        docker_service="safeear_detection",
        container_name="deepsafe-safeear-detection",
        architecture="SafeEar SpeechTokenizer + neural codec",
        paper="NeurIPS 2024",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="audio_data",
    ),
    ModelDef(
        name="sonics",
        modality="audio",
        port=8003,
        docker_service="sonics_detection",
        container_name="deepsafe-sonics-detection",
        architecture="SONICS SpecTTTra spectral transformer",
        paper="ICLR 2025",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="audio_data",
    ),
    ModelDef(
        name="nes2net",
        modality="audio",
        port=8004,
        docker_service="nes2net_detection",
        container_name="deepsafe-nes2net-detection",
        architecture="Nes2Net XLS-R 300M + nested Res2Net-TDNN",
        paper="IEEE T-IFS 2025",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="audio_data",
    ),

    # ── Provenance (4) ────────────────────────────────────────────────────
    ModelDef(
        name="c2pa",
        modality="provenance",
        port=9001,
        docker_service="c2pa-checker",
        container_name="deepsafe-c2pa-checker",
        architecture="C2PA Content Credentials manifest parser",
        paper="C2PA Spec 2.0",
        cuda_tier=0,
        mem_limit="512m",
        payload_key="image_data",
    ),
    ModelDef(
        name="sdxl_wm",
        modality="provenance",
        port=9002,
        docker_service="sdxl-watermark-detector",
        container_name="deepsafe-sdxl-watermark-detector",
        architecture="SDXL invisible watermark DWT extractor",
        paper="N/A",
        cuda_tier=0,
        mem_limit="1g",
        payload_key="image_data",
    ),
    ModelDef(
        name="audioseal",
        modality="provenance",
        port=9003,
        docker_service="audioseal-detector",
        container_name="deepsafe-audioseal-detector",
        architecture="AudioSeal localized audio watermark detector",
        paper="ICML 2024",
        cuda_tier=3,
        mem_limit="2g",
        payload_key="audio_data",
    ),
    ModelDef(
        name="videoseal",
        modality="provenance",
        port=9004,
        docker_service="videoseal-detector",
        container_name="deepsafe-videoseal-detector",
        architecture="VideoSeal frame-level video watermark detector",
        paper="Meta 2024",
        cuda_tier=3,
        mem_limit="4g",
        payload_key="video_data",
    ),
]

# Fast lookup by name.
_MODEL_MAP = {m.name: m for m in ALL_MODELS}

# Provenance services relevant to each detection modality.
# Matches the routing in deepsafe_config.json.
_PROVENANCE_FOR_MODALITY = {
    "image": {"c2pa", "sdxl_wm", "videoseal"},
    "video": {"c2pa", "videoseal"},
    "audio": {"c2pa", "audioseal"},
}


def get_models_by_modality(modality: str) -> List[ModelDef]:
    """Return all models for the given modality.

    Args:
        modality: One of "image", "video", "audio", or "provenance".

    Returns:
        List of matching ModelDef instances.
    """
    return [m for m in ALL_MODELS if m.modality == modality]


def get_model(name: str) -> Optional[ModelDef]:
    """Look up a single model by short name.

    Args:
        name: Short model name (e.g. "npr", "fakestormer").

    Returns:
        The ModelDef if found, else None.
    """
    return _MODEL_MAP.get(name)


def get_detection_models() -> List[ModelDef]:
    """Return the 19 detection models (excludes provenance services)."""
    return [m for m in ALL_MODELS if m.modality != "provenance"]


def get_docker_services(modality: str) -> List[ModelDef]:
    """Return detection + relevant provenance services for a modality.

    For benchmarking a full modality pipeline we need the detection
    models *and* the provenance services the gateway routes to for
    that modality.

    Args:
        modality: One of "image", "video", or "audio".

    Returns:
        Detection models for the modality plus their provenance
        dependencies, sorted by port number.
    """
    detection = get_models_by_modality(modality)
    provenance_names = _PROVENANCE_FOR_MODALITY.get(modality, set())
    provenance = [m for m in ALL_MODELS if m.name in provenance_names]
    combined = detection + provenance
    return sorted(combined, key=lambda m: m.port)
