"""
Configuration loading for the DeepSafe API gateway.

Loads model configuration from the JSON file specified by
``DEEPSAFE_CONFIG_FILE_PATH``, and defines all media-type constants used
throughout the gateway.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- Constants for Payload Keys and Media Handling ---
MEDIA_TYPE_PAYLOAD_KEYS: Dict[str, str] = {
    "image": "image_data",
    "video": "video_data",
    "audio": "audio_data",
}

MAX_IMAGE_SIZE_MB: int = 100
MAX_IMAGE_SIZE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_GENERAL_PAYLOAD_SIZE_BYTES: int = (MAX_IMAGE_SIZE_MB + 15) * 1024 * 1024

CONTENT_TYPE_TO_MEDIA_TYPE_MAP: Dict[str, str] = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
    "image/bmp": "image",
    "image/tiff": "image",
    "video/mp4": "video",
    "video/x-m4v": "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
    "video/x-matroska": "video",
    "video/webm": "video",
    "video/ogg": "video",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/wave": "audio",
    "audio/vnd.wave": "audio",
    "audio/mpeg": "audio",
    "audio/flac": "audio",
    "audio/ogg": "audio",
    "audio/x-m4a": "audio",
}

# Map content-type → file extension for MinIO object paths.
CONTENT_TYPE_TO_EXT: Dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "video/mp4": "mp4",
    "video/x-m4v": "m4v",
    "video/quicktime": "mov",
    "video/x-msvideo": "avi",
    "video/x-matroska": "mkv",
    "video/webm": "webm",
    "video/ogg": "ogv",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/vnd.wave": "wav",
    "audio/mpeg": "mp3",
    "audio/flac": "flac",
    "audio/ogg": "oga",
    "audio/x-m4a": "m4a",
}


def get_environment_variable(
    name: str, default: Optional[str] = None, required: bool = False
) -> Optional[str]:
    """Read an environment variable with optional default and required checks.

    Args:
        name: Name of the environment variable.
        default: Value to return if the variable is not set.
        required: If True, raises ValueError when the variable is missing.

    Returns:
        The variable value, the default, or None.

    Raises:
        ValueError: If required is True and the variable is not set.
    """
    value = os.environ.get(name)
    if value is None:
        if required:
            logger.error(f"FATAL: Required environment variable {name} not set.")
            raise ValueError(f"Missing required environment variable: {name}")
        if default is not None:
            logger.warning("Environment variable '%s' not set, using default.", name)
            return default
        return None
    return value


# --- Configuration Loading ---
ALL_MODEL_CONFIGS: Dict[str, Any] = {}
SUPPORTED_MEDIA_TYPES: List[str] = []

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "deepsafe_config.json"
)
CONFIG_FILE_PATH_FROM_ENV = get_environment_variable(
    "DEEPSAFE_CONFIG_FILE_PATH", default=_DEFAULT_CONFIG_PATH
)

if CONFIG_FILE_PATH_FROM_ENV and os.path.exists(CONFIG_FILE_PATH_FROM_ENV):
    logger.info(f"Loading configuration from: {CONFIG_FILE_PATH_FROM_ENV}")
    try:
        with open(CONFIG_FILE_PATH_FROM_ENV, "r") as f_config:
            loaded_json = json.load(f_config)
        ALL_MODEL_CONFIGS = loaded_json
        SUPPORTED_MEDIA_TYPES = list(ALL_MODEL_CONFIGS.get("media_types", {}).keys())
        if not SUPPORTED_MEDIA_TYPES:
            logger.error(
                f"FATAL: 'media_types' key missing or empty in {CONFIG_FILE_PATH_FROM_ENV}."
            )
            ALL_MODEL_CONFIGS = {"media_types": {}}
        else:
            logger.info(f"Active media types: {SUPPORTED_MEDIA_TYPES}")
    except json.JSONDecodeError as e:
        logger.error(f"FATAL: Malformed JSON in config file: {e}.")
        ALL_MODEL_CONFIGS = {"media_types": {}}
    except Exception as e:
        logger.error(f"FATAL: Config load failure: {e}.")
        ALL_MODEL_CONFIGS = {"media_types": {}}
else:
    logger.error(
        f"FATAL: Configuration file not found at '{CONFIG_FILE_PATH_FROM_ENV}'. Service cannot start."
    )
    ALL_MODEL_CONFIGS = {"media_types": {}}

DEFAULT_TIMEOUT: int = int(
    os.getenv(
        "DEEPSAFE_DEFAULT_TIMEOUT",
        str(ALL_MODEL_CONFIGS.get("default_api_timeout_seconds", 1200)),
    )
)
MAX_RETRIES: int = int(
    os.getenv(
        "DEEPSAFE_MAX_RETRIES",
        str(ALL_MODEL_CONFIGS.get("default_max_retries", 1)),
    )
)
