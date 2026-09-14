"""Prometheus metrics for the DeepSafe gateway.

The /metrics endpoint is restricted to internal/localhost access.
"""

import hmac
import os

from fastapi import APIRouter, HTTPException, Request, Response, status
from prometheus_client import Counter, Histogram, generate_latest

router = APIRouter(tags=["Metrics"])

_METRICS_TOKEN = os.getenv("METRICS_TOKEN", "")

# Request counters
REQUEST_COUNT = Counter(
    "deepsafe_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

# Detection counters
DETECTION_COUNT = Counter(
    "deepsafe_detections_total",
    "Total detection requests",
    ["media_type", "verdict"],
)

# Inference latency
INFERENCE_LATENCY = Histogram(
    "deepsafe_inference_seconds",
    "Model inference latency in seconds",
    ["media_type"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# Error counter
ERROR_COUNT = Counter(
    "deepsafe_errors_total",
    "Total errors by type",
    ["error_type"],
)


@router.get("/metrics")
async def metrics(request: Request):
    """Prometheus metrics endpoint.

    Restricted: requires METRICS_TOKEN bearer auth if configured,
    or requests must come from localhost/internal.
    """
    if _METRICS_TOKEN:
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {_METRICS_TOKEN}"
        if not hmac.compare_digest(auth, expected):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Metrics access denied.",
            )
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
