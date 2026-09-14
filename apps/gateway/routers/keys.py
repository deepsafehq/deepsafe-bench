"""API key management and dashboard endpoints."""

import logging
from datetime import datetime, timezone
from typing import Optional

from api_keys import ApiKey, generate_api_key, hash_key
from auth import get_current_user
from auth_helpers import extract_user_id
from database import AnalysisHistory, get_db
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from rate_limiter import TIER_LIMITS
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter()


class _CreateKeyRequest(BaseModel):
    """Request body for creating a new API key."""

    name: str = Field(..., min_length=1, max_length=100)


@router.post("/api/keys", status_code=201, tags=["API Keys"])
async def create_api_key(
    body: _CreateKeyRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new API key for the authenticated user.

    The plaintext key is returned only once and is not stored.

    Args:
        body: JSON body containing the desired key name.
        current_user: Decoded JWT payload injected by the auth dependency.
        db: Database session injected by FastAPI dependency.

    Returns:
        Dict with id, plaintext key, key_prefix, name, tier, and created_at.
    """
    user_id = extract_user_id(current_user)

    active_count = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user_id, ApiKey.revoked_at.is_(None))
        .count()
    )
    if active_count >= 5:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "key_limit",
                "message": "Maximum of 5 API keys allowed.",
            },
        )

    plaintext = generate_api_key()
    key_hash = hash_key(plaintext)
    key_prefix = plaintext[:16]
    api_key = ApiKey(
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=body.name,
        tier="free",
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    return {
        "id": str(api_key.id),
        "key": plaintext,
        "key_prefix": api_key.key_prefix,
        "name": api_key.name,
        "tier": api_key.tier,
        "created_at": api_key.created_at.isoformat(),
    }


@router.get("/api/keys", tags=["API Keys"])
async def list_api_keys(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all active (non-revoked) API keys for the authenticated user.

    Args:
        current_user: Decoded JWT payload injected by the auth dependency.
        db: Database session injected by FastAPI dependency.

    Returns:
        List of key summary dicts (plaintext key is never returned).
    """
    user_id = extract_user_id(current_user)
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user_id, ApiKey.revoked_at.is_(None))
        .all()
    )
    return [
        {
            "id": str(k.id),
            "key_prefix": k.key_prefix,
            "name": k.name,
            "tier": k.tier,
            "scans_used": k.scans_used,
            "created_at": k.created_at.isoformat(),
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
        }
        for k in keys
    ]


@router.delete("/api/keys/{key_id}", tags=["API Keys"])
async def revoke_api_key(
    key_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Soft-delete (revoke) an API key owned by the authenticated user.

    Args:
        key_id: UUID of the key to revoke.
        current_user: Decoded JWT payload injected by the auth dependency.
        db: Database session injected by FastAPI dependency.

    Returns:
        Dict with status "revoked".

    Raises:
        HTTPException: 404 if the key does not exist or belongs to another user.
    """
    user_id = extract_user_id(current_user)
    key = (
        db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == user_id).first()
    )
    if key is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": "API key not found."},
        )
    key.revoked_at = datetime.now(timezone.utc)
    db.commit()
    return {"status": "revoked"}


@router.post("/api/keys/auto", tags=["API Keys"])
async def auto_create_api_key(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Idempotently create a default API key for the authenticated user.

    On first call a new key named "Default" with tier "free" is created and the
    plaintext key is returned.  On subsequent calls only key metadata is returned
    — the plaintext is never stored and therefore cannot be returned again.

    Args:
        current_user: Decoded JWT payload injected by the auth dependency.
        db: Database session injected by FastAPI dependency.

    Returns:
        Dict containing ``created`` bool, and key metadata.  The plaintext
        ``key`` field is only present when ``created`` is ``True``.
    """
    user_id = extract_user_id(current_user)
    existing = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user_id, ApiKey.revoked_at.is_(None))
        .first()
    )
    if existing is not None:
        tier_config = TIER_LIMITS.get(existing.tier, TIER_LIMITS["free"])
        return {
            "created": False,
            "key_prefix": existing.key_prefix,
            "id": str(existing.id),
            "tier": existing.tier,
            "scans_used": existing.scans_used,
            "scans_limit": tier_config["monthly"],
        }

    plaintext = generate_api_key()
    key_hash = hash_key(plaintext)
    key_prefix = plaintext[:16]
    api_key = ApiKey(
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name="Default",
        tier="free",
        scans_used=0,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    tier_config = TIER_LIMITS.get("free", {"monthly": 200})
    return {
        "created": True,
        "key": plaintext,
        "key_prefix": api_key.key_prefix,
        "id": str(api_key.id),
        "tier": api_key.tier,
        "scans_used": api_key.scans_used,
        "scans_limit": tier_config["monthly"],
    }


@router.get("/api/dashboard", tags=["Dashboard"])
async def get_dashboard(
    current_user: dict = Depends(get_current_user),
    offset: int = 0,
    limit: int = 10,
    media_type: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return plan info and paginated detections for the authenticated user.

    Args:
        current_user: Decoded JWT payload injected by the auth dependency.
        offset: Number of detections to skip (for pagination).
        limit: Max detections to return (default 10, max 50).
        media_type: Optional filter — 'image', 'audio', or 'video'.
        db: Database session injected by FastAPI dependency.

    Returns:
        Dict with ``plan``, ``detections``, ``total_detections``, ``offset``, ``limit``.
    """
    user_id = extract_user_id(current_user)
    limit = min(limit, 50)
    api_key = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user_id, ApiKey.revoked_at.is_(None))
        .first()
    )

    if api_key is not None:
        tier = api_key.tier
        from routers.v1 import _get_user_total_scans, _maybe_reset_monthly_quota

        _maybe_reset_monthly_quota(api_key, db)
        scans_used = _get_user_total_scans(user_id, db)
    else:
        tier = "free"
        scans_used = 0

    tier_config = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    plan = {
        "tier": tier,
        "scans_used": scans_used,
        "scans_limit": tier_config["monthly"],
        "is_monthly": tier_config.get("is_monthly", False),
    }

    query = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id)
    if media_type and media_type in ("image", "audio", "video"):
        query = query.filter(AnalysisHistory.media_type == media_type)

    total_detections = query.count()

    detections = (
        query.order_by(AnalysisHistory.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    recent_detections = [
        {
            "id": str(d.request_id),
            "media_type": d.media_type,
            "verdict": d.verdict,
            "confidence": d.confidence,
            "created_at": d.timestamp.isoformat() + "Z" if d.timestamp else None,
        }
        for d in detections
    ]

    return {
        "plan": plan,
        "detections": recent_detections,
        "total_detections": total_detections,
        "offset": offset,
        "limit": limit,
    }
