"""Health and info endpoints."""

import asyncio
import logging
from typing import Any, Dict

from config import ALL_MODEL_CONFIGS
from fastapi import APIRouter, Request
from services.models import check_model_health_api

logger = logging.getLogger(__name__)

router = APIRouter()


_PUBLIC_MEDIA_TYPES = {"image", "video", "audio"}


@router.get("/", tags=["Info"])
async def root_api():
    """Return API metadata and configured media types."""
    media_types_config = (
        ALL_MODEL_CONFIGS.get("media_types", {}) if ALL_MODEL_CONFIGS else {}
    )
    configured_media_types = [
        mt for mt in media_types_config if mt in _PUBLIC_MEDIA_TYPES
    ]
    return {
        "name": "DeepSafe API",
        "version": "1.4.0",
        "message": "Welcome to the DeepSafe deepfake detection API.",
        "supported_media_types": configured_media_types,
        "documentation": "https://deepsafehq.github.io/deepsafe-bench/docs",
    }


@router.get("/health", tags=["System"])
async def health_check_api_endpoint(request: Request):
    """Check health of the gateway and all configured model microservices.

    Returns only aggregate status per media type — never exposes model
    counts, names, or infrastructure details.
    """
    req_id = request.state.request_id
    logger.info(f"Request {req_id}: Received main API health check.")

    system_health_report: Dict[str, Any] = {
        "status": "healthy",
        "media_types": {},
    }
    overall_system_is_healthy = True

    media_types_in_config = [
        mt
        for mt in (
            ALL_MODEL_CONFIGS.get("media_types", {}).keys() if ALL_MODEL_CONFIGS else []
        )
        if mt in _PUBLIC_MEDIA_TYPES
    ]

    loop = asyncio.get_event_loop()

    for m_type in media_types_in_config:
        all_models_for_type_healthy = True

        current_media_type_config = (
            ALL_MODEL_CONFIGS.get("media_types", {}) if ALL_MODEL_CONFIGS else {}
        ).get(m_type, {})
        model_endpoints_for_this_type = current_media_type_config.get(
            "model_endpoints", {}
        )

        if not model_endpoints_for_this_type:
            system_health_report["media_types"][m_type] = "unavailable"
        else:
            model_names = list(model_endpoints_for_this_type.keys())
            health_results = await asyncio.gather(
                *[
                    loop.run_in_executor(
                        None, check_model_health_api, model_name, m_type
                    )
                    for model_name in model_names
                ]
            )
            for model_health_info in health_results:
                if model_health_info.get("status") not in ("healthy", "ok"):
                    all_models_for_type_healthy = False

            if not all_models_for_type_healthy:
                overall_system_is_healthy = False

            system_health_report["media_types"][m_type] = (
                "healthy" if all_models_for_type_healthy else "degraded"
            )

    if not overall_system_is_healthy:
        system_health_report["status"] = "degraded"

    return system_health_report
