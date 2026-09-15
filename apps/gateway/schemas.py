"""Pydantic V2 response models for the DeepSafe public API."""

from typing import Optional

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response shape for all API errors."""

    error: str
    message: str


class DetectionResult(BaseModel):
    """Synchronous detection result returned by POST /v1/detect."""

    id: str
    verdict: str
    confidence: float
    media_type: str
    created_at: str


class AsyncDetectionAccepted(BaseModel):
    """Async acceptance response returned by POST /v1/detect when async=true."""

    id: str
    status: str = "processing"
    poll_url: str


class DetectionStatus(BaseModel):
    """Job status response returned by GET /v1/results/{detection_id}."""

    id: str
    status: str  # processing | complete | failed
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    media_type: Optional[str] = None
    poll_url: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class UsageResponse(BaseModel):
    """Scan usage and quota returned by GET /v1/usage.

    ``scans_limit`` and ``scans_remaining`` are null on an unmetered plan,
    which is what a self-hosted install running without auth uses. Null means
    unlimited, not zero.
    """

    plan: str
    scans_used: int
    scans_limit: Optional[int] = None
    scans_remaining: Optional[int] = None
    period_start: Optional[str] = None


class ApiKeyCreated(BaseModel):
    """Response when a new API key is created."""

    id: str
    key: str
    key_prefix: str
    name: str
    tier: str
    created_at: str


class ApiKeyInfo(BaseModel):
    """API key metadata (no plaintext key)."""

    id: str
    key_prefix: str
    name: str
    tier: str
    created_at: str
    last_used_at: Optional[str] = None


class HealthStatus(BaseModel):
    """Health check response returned by GET /health."""

    overall_api_status: str
    media_type_details: dict
    request_id: str
    processing_mode: str
