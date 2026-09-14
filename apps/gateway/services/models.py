"""Model health-checking utilities."""

import json
import logging
from typing import Any, Dict

import requests
from config import ALL_MODEL_CONFIGS

logger = logging.getLogger(__name__)


def check_model_health_api(model_name: str, media_type: str) -> Dict[str, Any]:
    """Check the health of a single model microservice.

    Args:
        model_name: Name of the model (key in config).
        media_type: Media category ('image', 'video', 'audio').

    Returns:
        Dict with 'status' key and optional 'message'.
    """
    media_type_config = ALL_MODEL_CONFIGS.get("media_types", {}).get(media_type, {})
    health_endpoints_for_type = media_type_config.get("health_endpoints", {})
    if model_name not in health_endpoints_for_type:
        return {
            "status": "error",
            "message": f"No health endpoint configured for model '{model_name}' of type '{media_type}'.",
        }

    health_url = health_endpoints_for_type[model_name]
    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.warning(
            f"Health check failed for {model_name} ({media_type}) at {health_url}: {e}"
        )
        return {"status": "unreachable", "message": str(e)}
    except json.JSONDecodeError:
        logger.warning(
            f"Health check for {model_name} ({media_type}) returned non-JSON: {response.text[:100]}"
        )
        return {"status": "invalid_response", "message": "Non-JSON health response"}
