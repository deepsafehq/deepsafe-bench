#!/usr/bin/env python3
"""
DeepSafe API Gateway
====================

The central orchestration layer for the DeepSafe platform. Registers the
Celery detection task (must live here so Celery uses the name 'main.run_detection')
and wires together all routers, middleware, and application lifecycle.

Configuration is driven by ``deepsafe_config.json`` via the ``DEEPSAFE_CONFIG_FILE_PATH``
environment variable.
"""

import json
import logging
import os
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import analytics
import sentry_sdk
import uvicorn
from auth import get_current_user  # noqa: F401 — re-exported for test compatibility
from celery_app import celery_app
from config import (
    ALL_MODEL_CONFIGS,
    CONFIG_FILE_PATH_FROM_ENV,
    MAX_GENERAL_PAYLOAD_SIZE_BYTES,
    SUPPORTED_MEDIA_TYPES,
    get_environment_variable,
)
from database import SessionLocal, init_db
from dependencies import MINIO_JOBS_BUCKET, minio_client, redis_client
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rate_limiter import RateLimiter

# Re-exported for backwards-compatibility with tests that import from main.
# fmt: off
from services.detection import _enqueue_async_detection  # noqa: F401
from services.detection import _insert_cost_tracking  # noqa: F401
from services.detection import _insert_model_performance  # noqa: F401
from services.detection import _run_sync_detection  # noqa: F401
from services.detection import query_model_api  # noqa: F401
from services.detection import run_detection_task

# fmt: on

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Sentry ---
_sentry_dsn = os.getenv("SENTRY_DSN", "")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=0.2,
        send_default_pii=False,
        environment=os.getenv("DEEPSAFE_ENV", "production"),
    )
    logger.info("Sentry initialized.")
else:
    logger.warning("SENTRY_DSN not set — error tracking disabled.")

# Lock to serialize per-model Redis updates from concurrent ThreadPoolExecutor threads.
_redis_model_update_lock = threading.Lock()


# --- Celery Task ---
# The @celery_app.task decorator must stay in main.py so Celery registers the
# task under the name "main.run_detection" — the worker uses include=["main"].
# The body is delegated to run_detection_task() in services/detection.py.
@celery_app.task(bind=True, max_retries=0, name="main.run_detection")
def run_detection(
    self,
    job_id: str,
    media_type: str,
    ext: str,
    threshold: float,
    ensemble_method: str,
    models_list: Optional[List[str]],
    user_id: Optional[str] = None,
) -> None:
    """Celery task: run deepfake detection for a single job.

    See ``services.detection.run_detection_task`` for full implementation.

    Args:
        job_id: UUID of the job.
        media_type: "image" | "video" | "audio".
        ext: File extension (e.g. "jpg", "wav", "mp4").
        threshold: Fake probability threshold (0.0-1.0).
        ensemble_method: Ensemble strategy name.
        models_list: Model names to run, or None for all configured.
        user_id: Supabase UUID of the authenticated user, or None for anonymous.
    """
    run_detection_task(
        _redis_model_update_lock,
        job_id,
        media_type,
        ext,
        threshold,
        ensemble_method,
        models_list,
        user_id,
    )


# --- FastAPI Application ---
_is_production = os.getenv("DEEPSAFE_ENV", "production").lower() == "production"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the FastAPI app."""
    # --- Startup ---
    analytics.init_analytics()
    init_db()
    logger.info("Database initialized successfully")

    try:
        if not minio_client.bucket_exists(MINIO_JOBS_BUCKET):
            minio_client.make_bucket(MINIO_JOBS_BUCKET)
            logger.info(f"Created MinIO bucket: {MINIO_JOBS_BUCKET}")
        else:
            logger.info(f"MinIO bucket '{MINIO_JOBS_BUCKET}' already exists.")
    except Exception as e:
        logger.warning(f"MinIO bucket init failed (continuing): {e}")

    logger.info(f"Configured media types: {SUPPORTED_MEDIA_TYPES}")

    yield

    # --- Shutdown ---
    analytics.shutdown_analytics()


app = FastAPI(
    title="DeepSafe API",
    description="Enterprise-grade API for deepfake detection.",
    version="1.4.0",
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
    lifespan=lifespan,
)

# --- Router Includes ---
import analytics_routes as _analytics_routes_module  # noqa: E402
import metrics as _metrics_module  # noqa: E402
from routers import v1 as _v1_router_module  # noqa: E402
from routers.health import router as _health_router  # noqa: E402
from routers.history import router as _history_router  # noqa: E402
from routers.keys import router as _keys_router  # noqa: E402

app.include_router(_analytics_routes_module.router)
app.include_router(_v1_router_module.router)
app.include_router(_metrics_module.router)
app.include_router(_health_router)
app.include_router(_history_router)
app.include_router(_keys_router)

# Attach Redis client and rate limiter to app.state so routers can access them
# via request.app.state without creating circular imports.
app.state.redis_client = redis_client
app.state.rate_limiter = RateLimiter(redis_client)


# --- Middleware ---
_cors_origins = [
    "https://deepsafehq.github.io/deepsafe-bench",
    "http://localhost:3000",
    "https://deepsafehq.github.io/deepsafe-bench/docs",
]
if os.getenv("DEEPSAFE_ENV", "production") != "production":
    _cors_origins.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def request_id_and_size_limit_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    if request.method == "POST" and request.url.path in ("/v1/detect",):
        # Check Content-Length header (fast reject for honest clients).
        content_length_str = request.headers.get("content-length")
        if content_length_str:
            try:
                content_length = int(content_length_str)
                if content_length > MAX_GENERAL_PAYLOAD_SIZE_BYTES:
                    logger.warning(
                        "Request %s: Payload size %d exceeds limit %d for %s.",
                        request.state.request_id,
                        content_length,
                        MAX_GENERAL_PAYLOAD_SIZE_BYTES,
                        request.url.path,
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Request payload too large. Max size: {MAX_GENERAL_PAYLOAD_SIZE_BYTES / (1024*1024):.1f}MB",
                            "request_id": request.state.request_id,
                        },
                    )
            except ValueError:
                logger.warning(
                    "Request %s: Invalid Content-Length header: %s",
                    request.state.request_id,
                    content_length_str,
                )
        else:
            # No Content-Length header (chunked encoding).
            # Read and buffer the body with a size limit to prevent
            # unbounded memory consumption.
            body = b""
            async for chunk in request.stream():
                body += chunk
                if len(body) > MAX_GENERAL_PAYLOAD_SIZE_BYTES:
                    logger.warning(
                        "Request %s: Chunked payload exceeded limit %d.",
                        request.state.request_id,
                        MAX_GENERAL_PAYLOAD_SIZE_BYTES,
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Request payload too large. Max size: {MAX_GENERAL_PAYLOAD_SIZE_BYTES / (1024*1024):.1f}MB",
                            "request_id": request.state.request_id,
                        },
                    )
            # Store buffered body so downstream handlers can read it.
            request._body = body

    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    if not hasattr(request.state, "request_id"):
        request.state.request_id = str(uuid.uuid4())
    try:
        return await call_next(request)
    except HTTPException as e:
        logger.warning(
            f"Request {request.state.request_id}: HTTPException raised: Status {e.status_code}, Detail: {e.detail}"
        )
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail, "request_id": request.state.request_id},
        )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception(
            f"Request {request.state.request_id}: Unhandled internal exception: {str(e)}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred.",
                "request_id": request.state.request_id,
            },
        )


if __name__ == "__main__":
    if (
        not CONFIG_FILE_PATH_FROM_ENV
        or not os.path.exists(CONFIG_FILE_PATH_FROM_ENV)
        or not ALL_MODEL_CONFIGS.get("media_types")
    ):
        logger.critical(
            "FATAL: API configuration (DEEPSAFE_CONFIG_FILE_PATH) is missing, invalid, or does not define 'media_types'. API cannot start meaningfully."
        )
        sys.exit(1)

    port = int(get_environment_variable("PORT", "8000"))
    workers = int(get_environment_variable("WORKERS", "1"))
    log_level = get_environment_variable("LOG_LEVEL", "info").lower()

    logger.info(
        f"Starting DeepSafe API (v{app.version}) on port {port} with {workers} worker(s). Log level: {log_level}"
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
    )
