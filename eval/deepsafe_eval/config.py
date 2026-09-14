"""Central configuration for the DeepSafe evaluation framework."""

import os
from pathlib import Path

# Dataset paths -- resolves from project symlink or env var
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.getenv("DEEPSAFE_DATASET", _PROJECT_ROOT / "dataset"))
MASTER_EVAL = DATASET_ROOT / "master_eval"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# HTTP timeout for model calls (seconds)
MODEL_TIMEOUT = 300

# Max parallel workers for model fan-out
MAX_WORKERS = 4

# --- Active Model Endpoints ---
# Use localhost ports since eval scripts run from the host,
# not inside the Docker network.

IMAGE_MODELS = {
    "npr": "http://localhost:5001/predict",
    "universal": "http://localhost:5003/predict",
    "yermandy": "http://localhost:5002/predict",
    "aide": "http://localhost:5004/predict",
    "fsd": "http://localhost:5005/predict",
    "effort": "http://localhost:5006/predict",
    "cospy": "http://localhost:5007/predict",
}

VIDEO_MODELS = {
    "fakestormer": "http://localhost:7001/predict",
    "sbi": "http://localhost:7002/predict",
    "dfd_fcg": "http://localhost:7003/predict",
    "pwtf_dvd": "http://localhost:7005/predict",
    "lipfd": "http://localhost:7006/predict",
    "recce": "http://localhost:7007/predict",
    "mintime": "http://localhost:7008/predict",
}

AUDIO_MODELS = {
    "shiftyspeech": "http://localhost:8001/predict",
    "safeear": "http://localhost:8002/predict",
    "nes2net": "http://localhost:8004/predict",
}

PROVENANCE_MODELS = {
    "c2pa": "http://localhost:9001/predict",
    "sdxl_wm": "http://localhost:9002/predict",
    "audioseal": "http://localhost:9003/predict",
    "videoseal": "http://localhost:9004/predict",
}

# Modality -> (payload_key, models_dict)
MODALITY_CONFIG = {
    "images": ("image_data", IMAGE_MODELS),
    "audio": ("audio_data", AUDIO_MODELS),
    "video": ("video_data", VIDEO_MODELS),
}

# --- Generator Family Classification ---
GENERATOR_FAMILIES = {
    "gan": [
        "stylegan2",
        "stylegan3",
        "stargan",
        "starganv2",
        "styleganxl",
        "biggan",
        "vqgan",
    ],
    "diffusion": [
        "stable_diffusion_xl",
        "stable_diffusion_3",
        "stable_diffusion_2",
        "stable_diffusion_1_4",
        "stable_diffusion_1_3",
        "sd_1.5",
        "sd_2.1",
        "dalle_3",
        "dalle_2",
        "flux_1",
        "flux_schnell",
        "midjourney_v5",
        "midjourney_6",
        "midjourney_7",
        "firefly",
        "glide",
        "gemini",
        "imagen_3",
        "imagen_4.0",
        "gpt_image_1",
        "grok_2_image_1212",
        "ideogram_3.0",
        "recraft_v3",
        "chroma",
        "hidream_i1_full",
        "mystic",
    ],
    "face_manipulation": [
        "face_swap",
        "face_editing",
        "face_reenactment",
        "e4e",
        "styleclip",
        "whichfaceisreal",
    ],
    "vocoder": [
        "melgan",
        "waveglow",
        "hifigan",
        "griffin_lim",
        "parallel_wavegan",
        "multi_band_melgan",
        "full_band_melgan",
        "melgan_large",
        "neural_codec",
    ],
    "tts": [
        "tacotron",
        "wavenet",
        "conformer_fastspeech2_pwg",
        "bark",
        "elevenlabs",
        "valle",
        "xtts",
        "yourtts",
        "rvc",
    ],
    "music_gen": [
        "suno",
        "udio",
        "musicgen",
        "stable_audio",
        "riffusion",
    ],
    "t2v": [
        "sora",
        "gen2",
        "kling",
        "veo",
        "lavie",
        "crafter",
        "show1",
        "modelscope_gen",
        "morphstudio",
        "hotshot",
        "moonvalley",
        "hunyuan",
    ],
    "face_swap_video": [
        "faceswap",
        "deepfacelab",
        "face2face",
        "fsgan",
        "simswap",
        "facedancer",
        "wav2lip",
        "sadtalker",
    ],
}


def get_family(generator: str) -> str:
    """Return the family name for a generator, or 'unknown'.

    Args:
        generator: The generator name to classify.

    Returns:
        Family name string.
    """
    for family, members in GENERATOR_FAMILIES.items():
        if generator in members:
            return family
    return "unknown"
