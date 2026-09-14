"""History and job-status endpoints."""

import json
import logging
from typing import Optional

from auth import get_current_user
from auth_helpers import extract_user_id
from database import AnalysisHistory, get_db, get_db_session
from dependencies import redis_client
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter()

_VALID_MEDIA_TYPES = {"image", "video", "audio"}

# Fields that must never be returned to users — they expose internal
# model names, ensemble architecture, and processing details.
_INTERNAL_RESULT_KEYS = {
    "model_results",
    "ensemble_method_used",
    "processing_mode",
}


def _sanitize_result(result: dict) -> dict:
    """Strip internal model names and architecture details from a result.

    Args:
        result: Raw result payload from Redis or PostgreSQL.

    Returns:
        Filtered dict safe for user-facing responses.
    """
    return {k: v for k, v in result.items() if k not in _INTERNAL_RESULT_KEYS}


@router.get("/jobs/{job_id}", tags=["Web UI"])
async def get_job_status(job_id: str, current_user: dict = Depends(get_current_user)):
    """Return current status of an async detection job.

    Checks Redis first; falls back to PostgreSQL for expired keys.
    Returns 404 if the job is not found in either store.
    Only returns jobs belonging to the authenticated user.
    """
    user_id = extract_user_id(current_user)
    redis_key = f"job:{job_id}"
    raw = redis_client.get(redis_key)

    if raw:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Job %s: Corrupt Redis data, skipping.", job_id)
            data = None

        if data and data.get("user_id") != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "not_found", "message": f"Job '{job_id}' not found."},
            )
        if data:
            safe_keys = {
                "job_id",
                "status",
                "media_type",
                "created_at",
                "completed_at",
                "result",
                "error",
            }
            filtered = {k: v for k, v in data.items() if k in safe_keys}
            # Sanitize the result dict to strip internal model/ensemble
            # details.
            if isinstance(filtered.get("result"), dict):
                filtered["result"] = _sanitize_result(filtered["result"])
            return filtered

    # Redis TTL expired — try PostgreSQL fallback.
    with get_db_session() as db:
        record = (
            db.query(AnalysisHistory)
            .filter(AnalysisHistory.request_id == job_id)
            .filter(AnalysisHistory.user_id == user_id)
            .first()
        )
        if record:
            try:
                result_payload = (
                    json.loads(record.full_response) if record.full_response else {}
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning("Job %s: Corrupt full_response in DB.", job_id)
                result_payload = {}
            return {
                "job_id": job_id,
                "status": "COMPLETE",
                "media_type": record.media_type,
                "created_at": record.timestamp.isoformat(),
                "completed_at": record.timestamp.isoformat(),
                "result": _sanitize_result(result_payload),
                "error": None,
            }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": f"Job '{job_id}' not found."},
    )


@router.get("/history", tags=["History"])
async def get_analysis_history(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    media_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Retrieve analysis history with pagination and filtering."""
    user_id = extract_user_id(current_user)
    query = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id)
    if media_type:
        if media_type not in _VALID_MEDIA_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "invalid_media_type",
                    "message": (
                        f"Invalid media_type '{media_type}'. "
                        f"Valid values: {', '.join(sorted(_VALID_MEDIA_TYPES))}."
                    ),
                },
            )
        query = query.filter(AnalysisHistory.media_type == media_type)

    total = query.count()
    records = (
        query.order_by(AnalysisHistory.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": [
            {
                "id": r.id,
                "request_id": r.request_id,
                "media_type": r.media_type,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in records
        ],
    }


@router.get("/history/{request_id}", tags=["History"])
async def get_analysis_by_id(
    request_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Retrieve a specific analysis result by request ID."""
    user_id = extract_user_id(current_user)
    record = (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.request_id == request_id)
        .filter(AnalysisHistory.user_id == user_id)
        .first()
    )
    if not record:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": "Analysis not found."},
        )

    return {
        "id": record.id,
        "request_id": record.request_id,
        "media_type": record.media_type,
        "verdict": record.verdict,
        "confidence": record.confidence,
        "timestamp": record.timestamp.isoformat(),
    }
