"""Public /v1/ API router with dual authentication.

Supports two auth methods via the Bearer token:
- **API key** (``ds_live_...``): looked up by hash — used by external developers.
- **JWT** (Supabase session token): verified via JWKS/HS256 — used by the web UI.

For JWT auth the user's first active API key is used for billing/quota.
If no key exists one is auto-created.

Exposes three endpoints:
- POST /v1/detect   — submit media for deepfake detection (sync or async)
- GET  /v1/results/{detection_id} — poll async job result
- GET  /v1/usage    — current scan usage and quota for the authenticated key
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from api_keys import KEY_PREFIX, ApiKey, generate_api_key, hash_key
from config import LOCAL_USER_ID, REQUIRE_AUTH
from database import AnalysisHistory, SessionLocal, get_db
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from rate_limiter import TIER_LIMITS, RateLimitExceeded
from schemas import DetectionResult, DetectionStatus, UsageResponse  # noqa: F401
from sqlalchemy import func

logger = logging.getLogger(__name__)

# Stand-in token used when DEEPSAFE_REQUIRE_AUTH is off.
LOCAL_TOKEN = "local"

router = APIRouter(prefix="/v1", tags=["Public API v1"])

# Prefix applied to all detection IDs returned by this API.
_DET_ID_PREFIX = "det_"

# -----------------------------------------------------------------------
# Error response helper
# -----------------------------------------------------------------------


def _err(code: str, message: str, http_status: int) -> HTTPException:
    """Build an HTTPException whose detail matches the standard error shape.

    Args:
        code: Machine-readable error code (e.g. 'unauthorized').
        message: Human-readable description.
        http_status: HTTP status code to return.

    Returns:
        An HTTPException ready to raise.
    """
    return HTTPException(
        status_code=http_status,
        detail={"error": code, "message": message},
    )


# Detection ID format: det_ followed by 1-32 alphanumeric characters.
_DET_ID_PATTERN = re.compile(r"^det_[a-zA-Z0-9_]{1,32}$")

# File content magic bytes for basic validation.
_MAGIC_BYTES = {
    b"\xff\xd8\xff": "image",  # JPEG
    b"\x89PNG": "image",  # PNG
    b"RIFF": "image_or_audio",  # WebP or WAV (RIFF container)
    b"BM": "image",  # BMP
    b"II": "image",  # TIFF (little-endian)
    b"MM": "image",  # TIFF (big-endian)
    b"\x00\x00\x00": "video",  # MP4/MOV (ftyp box)
    b"\x1a\x45\xdf\xa3": "video",  # MKV/WebM (EBML)
    b"fLaC": "audio",  # FLAC
    b"OggS": "audio",  # OGG
    b"\xff\xfb": "audio",  # MP3 (frame sync)
    b"\xff\xf3": "audio",  # MP3 (frame sync)
    b"\xff\xf2": "audio",  # MP3 (frame sync)
    b"ID3": "audio",  # MP3 with ID3 tag
}


@dataclass
class AuthResult:
    """Result of authenticating a request — carries billing key and auth method."""

    api_key: ApiKey
    is_jwt_auth: bool


# -----------------------------------------------------------------------
# Auth dependency
# -----------------------------------------------------------------------


def _get_auth_header(request: Request) -> str:
    """Extract the Bearer token from the Authorization header.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The raw token string.

    Raises:
        HTTPException: 401 if the header is missing or malformed.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise _err("unauthorized", "Missing or malformed Authorization header.", 401)
    token = auth[len("Bearer ") :].strip()
    if not token:
        raise _err("unauthorized", "Bearer token is empty.", 401)
    return token


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------


def _lookup_api_key(token: str, db) -> ApiKey:
    """Hash token and return the matching ApiKey row.

    Args:
        token: Plaintext API key supplied by the caller.
        db: SQLAlchemy DB session.

    Returns:
        The matching, non-revoked ApiKey ORM instance.

    Raises:
        HTTPException: 401 if the key is unknown or has been revoked.
    """
    token_hash = hash_key(token)
    api_key = db.query(ApiKey).filter(ApiKey.key_hash == token_hash).first()
    if api_key is None or api_key.revoked_at is not None:
        raise _err("unauthorized", "Invalid or revoked API key.", 401)
    return api_key


def _resolve_local_api_key(db) -> ApiKey:
    """Return (or create) the billing record used when auth is disabled.

    Quota and rate limiting still run against this record, so the code path is
    the same one production uses. The tier is unlimited because metering a
    single self-hosted user against a SaaS plan would be meaningless.

    Args:
        db: SQLAlchemy DB session.

    Returns:
        The local ApiKey ORM instance.
    """
    api_key = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == LOCAL_USER_ID, ApiKey.revoked_at.is_(None))
        .first()
    )
    if api_key is not None:
        return api_key

    plaintext = generate_api_key()
    api_key = ApiKey(
        user_id=LOCAL_USER_ID,
        key_hash=hash_key(plaintext),
        key_prefix=plaintext[:16],
        name="Local (auth disabled)",
        tier="unlimited",
        scans_used=0,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    logger.warning(
        "Auth is disabled; serving requests as '%s'. Set "
        "DEEPSAFE_REQUIRE_AUTH=true before exposing this gateway to a network.",
        LOCAL_USER_ID,
    )
    return api_key


def _resolve_api_key_for_jwt_user(user_id: str, db) -> ApiKey:
    """Return (or auto-create) the billing ApiKey for a JWT-authenticated user.

    The web UI authenticates via Supabase JWT rather than an API key,
    but billing is still tracked through the ApiKey table.  This helper
    finds the user's first active key, or creates one if none exists.

    Args:
        user_id: Supabase user UUID (from the JWT ``sub`` claim).
        db: SQLAlchemy DB session.

    Returns:
        An active ApiKey ORM instance for billing purposes.
    """
    api_key = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user_id, ApiKey.revoked_at.is_(None))
        .first()
    )
    if api_key is not None:
        return api_key

    # First time this user hits the API — create a default key.
    plaintext = generate_api_key()
    api_key = ApiKey(
        user_id=user_id,
        key_hash=hash_key(plaintext),
        key_prefix=plaintext[:16],
        name="Default",
        tier="free",
        scans_used=0,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    return api_key


def _authenticate(token: str, db) -> AuthResult:
    """Resolve a Bearer token to an AuthResult for billing, supporting both auth methods.

    - Tokens starting with ``ds_live_`` are treated as API keys (hashed lookup).
    - All other tokens are verified as Supabase JWTs; the user's billing
      ApiKey is resolved (or created) from the JWT ``sub`` claim.

    Args:
        token: Raw Bearer token from the Authorization header.
        db: SQLAlchemy DB session.

    Returns:
        An AuthResult with the ApiKey and auth method flag.

    Raises:
        HTTPException: 401 on invalid key or JWT.
    """
    if not REQUIRE_AUTH and (token == LOCAL_TOKEN or not token):
        return AuthResult(api_key=_resolve_local_api_key(db), is_jwt_auth=True)

    if token.startswith(KEY_PREFIX):
        api_key = _lookup_api_key(token, db)
        return AuthResult(api_key=api_key, is_jwt_auth=False)

    # JWT path — web UI users get async access regardless of tier.
    from auth import verify_supabase_jwt

    try:
        payload = verify_supabase_jwt(token)
    except Exception:
        raise _err("unauthorized", "Invalid authentication token.", 401)

    user_id = payload.get("sub")
    if not user_id:
        raise _err("unauthorized", "Token missing user identifier.", 401)

    api_key = _resolve_api_key_for_jwt_user(user_id, db)
    return AuthResult(api_key=api_key, is_jwt_auth=True)


def _check_rate_limit(api_key: ApiKey, redis_client, limiter) -> None:
    """Enforce per-minute rate limit via Redis.

    Args:
        api_key: The authenticated ApiKey row.
        redis_client: Redis client instance.
        limiter: RateLimiter instance.

    Raises:
        HTTPException: 429 if the rate limit has been exceeded.
    """
    try:
        limiter.check(api_key.user_id, api_key.tier)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limited",
                "message": f"Rate limit exceeded. Retry after {exc.retry_after}s.",
                "retry_after": exc.retry_after,
            },
        ) from exc


def _maybe_reset_monthly_quota(api_key: ApiKey, db) -> None:
    """Reset scans_used for monthly tiers when the billing period has expired.

    For tiers with ``is_monthly=True`` (Starter, Pro), checks whether
    30 days have elapsed since ``period_start``.  If so, resets
    ``scans_used`` to 0 and updates ``period_start`` for ALL of the
    user's active keys so the quota pool stays consistent.

    Free-tier keys (``is_monthly=False``) are never reset.

    Args:
        api_key: The authenticated ApiKey row (mutated in-place on reset).
        db: SQLAlchemy DB session (committed on reset).
    """
    tier_config = TIER_LIMITS.get(api_key.tier, TIER_LIMITS["free"])
    if not tier_config.get("is_monthly", False):
        return
    if api_key.period_start is None:
        return

    now = datetime.now(timezone.utc)
    period_start = api_key.period_start
    if period_start.tzinfo is None:
        period_start = period_start.replace(tzinfo=timezone.utc)
    period_end = period_start + timedelta(days=30)
    if now < period_end:
        return

    # Reset all of this user's active keys so the aggregate stays correct.
    db.query(ApiKey).filter(
        ApiKey.user_id == api_key.user_id,
        ApiKey.revoked_at.is_(None),
    ).update({"scans_used": 0, "period_start": now})
    db.commit()
    api_key.scans_used = 0
    api_key.period_start = now


def _get_user_total_scans(user_id: str, db) -> int:
    """Sum scans_used across ALL of a user's active (non-revoked) keys.

    This ensures the quota is per-user, not per-key.  A user with
    multiple API keys cannot exceed their tier limit by spreading
    usage across keys.

    Args:
        user_id: Supabase user UUID.
        db: SQLAlchemy DB session.

    Returns:
        Total scans used by the user across all active keys.
    """
    total = (
        db.query(func.coalesce(func.sum(ApiKey.scans_used), 0))
        .filter(ApiKey.user_id == user_id, ApiKey.revoked_at.is_(None))
        .scalar()
    )
    return int(total)


def _check_monthly_quota(api_key: ApiKey, db) -> None:
    """Raise 402 if the user's aggregate scan quota has been reached.

    Checks the combined scans_used across ALL of the user's active keys
    against the tier limit.  For monthly tiers, resets first if the
    billing period has expired.

    Args:
        api_key: The authenticated ApiKey row.
        db: SQLAlchemy DB session.

    Raises:
        HTTPException: 402 if the caller has exhausted their quota.
    """
    _maybe_reset_monthly_quota(api_key, db)

    tier = api_key.tier
    limit = TIER_LIMITS.get(tier, {}).get("monthly")
    if limit is None:
        return

    total_scans = _get_user_total_scans(api_key.user_id, db)
    if total_scans >= limit:
        is_monthly = TIER_LIMITS.get(tier, {}).get("is_monthly", False)
        label = "Monthly" if is_monthly else "Lifetime"
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "quota_exceeded",
                "message": (
                    f"{label} quota of {limit} scans reached for '{tier}' plan. "
                    "Upgrade your plan to continue."
                ),
            },
        )


def _check_async_allowed(auth_result: AuthResult, is_async: bool) -> None:
    """Raise 402 if the caller requests async mode but their tier forbids it.

    JWT-authenticated requests (web UI) are always allowed async access
    regardless of tier, since the web demo needs it to avoid Cloudflare
    proxy timeouts on sync requests.

    Args:
        auth_result: The authenticated AuthResult.
        is_async: True if the caller wants async processing.

    Raises:
        HTTPException: 402 if async is not available on the caller's tier.
    """
    if not is_async:
        return
    # Web UI (JWT auth) always gets async access.
    if auth_result.is_jwt_auth:
        return
    if auth_result.api_key.tier == "free":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "async_not_allowed",
                "message": (
                    "Async processing is not available on the free tier. "
                    "Upgrade to Starter or Pro."
                ),
            },
        )


def _increment_usage(api_key: ApiKey, db) -> None:
    """Atomically increment scans_used and update last_used_at.

    Args:
        api_key: The authenticated ApiKey row (will be mutated in-place).
        db: SQLAlchemy DB session (will be committed).
    """
    db.query(ApiKey).filter(ApiKey.id == api_key.id).update(
        {
            "scans_used": ApiKey.scans_used + 1,
            "last_used_at": datetime.now(timezone.utc),
        }
    )
    db.commit()
    # Keep in-memory object consistent so callers see the new count.
    api_key.scans_used += 1
    api_key.last_used_at = datetime.now(timezone.utc)


def _validate_image_decodable(file_bytes: bytes) -> None:
    """Verify the image bytes can be opened by PIL.

    Catches corrupted or non-image files that pass magic-byte checks but
    fail in every model container, avoiding 7 redundant 400 errors.

    Raises:
        HTTPException: 400 if the file cannot be decoded as an image.
    """
    import io as _io

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(_io.BytesIO(file_bytes)) as img:
            img.verify()
    except (UnidentifiedImageError, SyntaxError, OSError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_image",
                "message": (
                    "The uploaded file could not be decoded as a valid image. "
                    "Please ensure the file is a valid JPEG, PNG, WebP, BMP, or TIFF."
                ),
            },
        )


def _validate_file_content(file_bytes: bytes, expected_media_type: str) -> None:
    """Validate file content magic bytes match the expected media type.

    This prevents clients from uploading arbitrary files (executables, zip
    bombs) with a spoofed Content-Type header.

    Args:
        file_bytes: Raw file bytes.
        expected_media_type: Expected category ('image', 'video', 'audio').

    Raises:
        HTTPException: 415 if magic bytes don't match expected type.
    """
    if len(file_bytes) < 4:
        return  # Too small to validate, let model services handle it.

    header = file_bytes[:12]
    matched = False
    for magic, media_category in _MAGIC_BYTES.items():
        if header.startswith(magic):
            if media_category == "image_or_audio":
                # RIFF container: could be WAV or WebP.
                matched = True
            elif media_category == expected_media_type:
                matched = True
            elif media_category != expected_media_type:
                # Magic bytes match a different media type.
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail={
                        "error": "content_type_mismatch",
                        "message": (
                            f"File content does not match declared type. "
                            f"Expected {expected_media_type}, "
                            f"but file appears to be {media_category}."
                        ),
                    },
                )
            break

    # If no magic bytes matched, allow it through — the model services
    # will reject truly invalid files. We only block clear mismatches.
    if not matched:
        return


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------


_SENSITIVITY_THRESHOLDS = {
    "balanced": 0.5,
    "strict": 0.7,
    "sensitive": 0.3,
}


def _require_auth(request: Request) -> str:
    """FastAPI dependency: extract Bearer token before body parsing.

    By using this as a ``Depends()`` parameter, FastAPI resolves it before
    attempting to parse the multipart body.  This ensures unauthenticated
    requests receive 401 rather than 422 (missing file).

    When auth is disabled (the default outside production) a sentinel token is
    returned instead, so a local install works without any credentials. A
    caller that does supply a header is still honoured.
    """
    if not REQUIRE_AUTH:
        auth = request.headers.get("Authorization", "")
        return auth[len("Bearer ") :].strip() if auth.startswith("Bearer ") else LOCAL_TOKEN
    return _get_auth_header(request)


@router.post("/detect", status_code=200)
async def v1_detect(
    request: Request,
    token: str = Depends(_require_auth),
    file: UploadFile = File(...),
    media_type: Optional[str] = Form(None),
    async_mode: Optional[bool] = Form(None, alias="async"),
    sensitivity: Optional[str] = Form(None),
    db=Depends(get_db),
):
    """Submit media for deepfake detection.

    Args:
        request: Incoming FastAPI request (carries app state + auth header).
        token: Bearer token extracted by _require_auth dependency.
        file: The media file to analyse.
        media_type: Optional hint for the media type ('image', 'video', 'audio').
        async_mode: If True, enqueue asynchronously and return 202. Sync by default.

    Returns:
        200 JSON with verdict and confidence (sync) or 202 JSON with job info (async).

    Raises:
        HTTPException: 401 if auth fails, 402 for quota/plan issues,
                       415 for unsupported MIME, 429 for rate limiting,
                       500 for internal errors.
    """
    # ---- Auth (API key or JWT) ----
    auth_result = _authenticate(token, db)
    api_key = auth_result.api_key

    # ---- Rate limit ----
    redis_client = request.app.state.redis_client
    limiter = request.app.state.rate_limiter
    _check_rate_limit(api_key, redis_client, limiter)

    # ---- Quota (per-user aggregate, with monthly reset) ----
    _check_monthly_quota(api_key, db)

    # ---- Async gate ----
    is_async = bool(async_mode)
    _check_async_allowed(auth_result, is_async)

    # ---- Resolve media type from MIME ----
    from config import (  # avoid circular at module level
        CONTENT_TYPE_TO_MEDIA_TYPE_MAP,
    )

    content_type = file.content_type or ""
    inferred_media_type = CONTENT_TYPE_TO_MEDIA_TYPE_MAP.get(content_type)
    if not inferred_media_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error": "unsupported_media_type",
                "message": (
                    f"Unsupported file type '{content_type}'. "
                    "Supported: JPEG, PNG, WebP, BMP, TIFF, MP4, MOV, AVI, MKV, "
                    "WebM, WAV, MP3, FLAC, OGG, M4A."
                ),
            },
        )

    # Use caller-supplied media_type hint only if it is valid.
    final_media_type = inferred_media_type

    # ---- Read file ----
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "empty_file", "message": "Uploaded file is empty."},
        )

    # ---- Validate file content matches claimed type ----
    _validate_file_content(file_bytes, final_media_type)

    # ---- Validate image can actually be decoded ----
    if final_media_type == "image":
        _validate_image_decodable(file_bytes)

    detection_id = _DET_ID_PREFIX + uuid.uuid4().hex[:8]

    if is_async:
        # ---- Async path ----
        # Usage is incremented by the Celery worker on successful completion.
        from services.detection import _enqueue_async_detection

        _enqueue_async_detection(
            file_bytes,
            content_type,
            final_media_type,
            api_key.user_id,
            detection_id,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "id": detection_id,
                "status": "processing",
                "poll_url": f"/v1/results/{detection_id}",
            },
        )

    # ---- Resolve sensitivity to threshold ----
    threshold_override = None
    if sensitivity:
        s = sensitivity.strip().lower()
        if s not in _SENSITIVITY_THRESHOLDS:
            raise _err(
                "invalid_sensitivity",
                f"Invalid sensitivity '{sensitivity}'. "
                f"Valid values: {', '.join(sorted(_SENSITIVITY_THRESHOLDS))}.",
                422,
            )
        threshold_override = _SENSITIVITY_THRESHOLDS[s]

    # ---- Sync path ----
    from services.detection import _run_sync_detection

    result = _run_sync_detection(
        file_bytes,
        content_type,
        final_media_type,
        api_key.user_id,
        detection_id,
        db,
        threshold_override=threshold_override,
    )

    # Increment usage only after successful detection.
    _increment_usage(api_key, db)

    return {
        "id": detection_id,
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "media_type": final_media_type,
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
    }


@router.get("/results/{detection_id}", response_model=DetectionStatus)
async def v1_results(detection_id: str, request: Request, db=Depends(get_db)):
    """Poll the result of an async detection job.

    Args:
        detection_id: The detection ID returned by POST /v1/detect.
        request: Incoming FastAPI request (carries app state + auth header).
        db: Database session (injected).

    Returns:
        JSON with status ('processing' | 'complete' | 'failed') and, when
        complete, the verdict and confidence.

    Raises:
        HTTPException: 401 if auth fails, 404 if not found.
    """
    # ---- Auth (must run before format validation to avoid leaking 404) ----
    token = _require_auth(request)
    auth_result = _authenticate(token, db)
    api_key = auth_result.api_key

    # ---- Validate detection_id format ----
    if not _DET_ID_PATTERN.match(detection_id):
        raise _err("not_found", f"Detection '{detection_id}' not found.", 404)

    redis_client = request.app.state.redis_client

    redis_key = f"job:{detection_id}"
    raw = redis_client.get(redis_key)

    if raw:
        data = json.loads(raw)
        if data.get("user_id") != api_key.user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "not_found",
                    "message": f"Detection '{detection_id}' not found.",
                },
            )
        return _shape_result_response(detection_id, data)

    # ---- Redis TTL expired — try PostgreSQL fallback ----
    record = (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.request_id == detection_id)
        .filter(AnalysisHistory.user_id == api_key.user_id)
        .first()
    )
    if record:
        return {
            "id": detection_id,
            "status": "complete",
            "verdict": record.verdict,
            "confidence": record.confidence,
            "media_type": record.media_type,
            "created_at": record.timestamp.isoformat() + "Z",
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error": "not_found",
            "message": f"Detection '{detection_id}' not found.",
        },
    )


def _shape_result_response(detection_id: str, data: dict) -> dict:
    """Convert internal Redis job state to public API response shape.

    Args:
        detection_id: The detection ID (with det_ prefix).
        data: Raw job state dict from Redis.

    Returns:
        A dict ready to be serialised as JSON.
    """
    raw_status = data.get("status", "PROCESSING")

    if raw_status == "COMPLETE":
        inner = data.get("result", {})
        return {
            "id": detection_id,
            "status": "complete",
            "verdict": inner.get("verdict")
            or ("fake" if inner.get("is_likely_deepfake") else "real"),
            "confidence": inner.get("confidence") or inner.get("deepfake_probability"),
            "media_type": inner.get("media_type") or inner.get("media_type_processed"),
            "created_at": data.get("created_at"),
            "completed_at": data.get("completed_at"),
        }

    if raw_status == "FAILED":
        return {
            "id": detection_id,
            "status": "failed",
            "error": data.get("error"),
            "created_at": data.get("created_at"),
        }

    # QUEUED or PROCESSING
    return {
        "id": detection_id,
        "status": "processing",
        "poll_url": f"/v1/results/{detection_id}",
        "created_at": data.get("created_at"),
    }


@router.get("/usage", response_model=UsageResponse)
async def v1_usage(request: Request, db=Depends(get_db)):
    """Return current scan usage and quota for the authenticated API key.

    Args:
        request: Incoming FastAPI request (carries auth header).
        db: Database session (injected).

    Returns:
        JSON with plan, scans_used, scans_limit, scans_remaining, period_start.

    Raises:
        HTTPException: 401 if auth fails.
    """
    token = _require_auth(request)
    auth_result = _authenticate(token, db)
    api_key = auth_result.api_key

    redis_client = request.app.state.redis_client
    limiter = request.app.state.rate_limiter
    _check_rate_limit(api_key, redis_client, limiter)

    _maybe_reset_monthly_quota(api_key, db)

    tier = api_key.tier
    # None means unmetered (the self-hosted tier), not zero.
    monthly_limit = TIER_LIMITS.get(tier, {}).get("monthly")
    scans_used = _get_user_total_scans(api_key.user_id, db)
    scans_remaining = (
        None if monthly_limit is None else max(monthly_limit - scans_used, 0)
    )

    return {
        "plan": tier,
        "scans_used": scans_used,
        "scans_limit": monthly_limit,
        "scans_remaining": scans_remaining,
        "period_start": (
            api_key.period_start.isoformat() + "Z" if api_key.period_start else None
        ),
    }
