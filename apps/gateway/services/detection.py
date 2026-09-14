"""
Core detection orchestration for the DeepSafe gateway.

Provides:
- ``query_model_api``: fan-out to a single model microservice.
- ``_insert_model_performance``: persist per-model results to PostgreSQL.
- ``_insert_cost_tracking``: persist cost/quota tracking row to PostgreSQL.
- ``_run_sync_detection``: synchronous detection path used by the public v1 API.
- ``_enqueue_async_detection``: async path — upload to MinIO and enqueue Celery.
"""

import base64
import concurrent.futures
import io
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import analytics
import requests
from config import (
    ALL_MODEL_CONFIGS,
    CONTENT_TYPE_TO_EXT,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    MEDIA_TYPE_PAYLOAD_KEYS,
)
from database import (
    AnalysisHistory,
    CostTracking,
    ModelPerformance,
    SessionLocal,
    get_db_session,
)
from deepsafe_shared.ensemble import (
    PROVENANCE_OVERRIDE_THRESHOLD,
    PROVENANCE_SERVICES,
    calculate_ensemble_verdict_api,
)
from dependencies import MINIO_JOBS_BUCKET, minio_client, redis_client
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


def query_model_api(
    model_name: str,
    media_type: str,
    encoded_media_data: str,
    threshold: float,
    request_id: str,
) -> Dict[str, Any]:
    """Send media to a single model microservice and return its result.

    Args:
        model_name: Name of the model (key in config).
        media_type: Media category ('image', 'video', 'audio').
        encoded_media_data: Base64-encoded media bytes.
        threshold: Fake probability threshold (0.0–1.0).
        request_id: Request/job ID used in log messages.

    Returns:
        Dict with model output on success, or ``{'error': ...}`` on failure.
    """
    media_type_config = ALL_MODEL_CONFIGS.get("media_types", {}).get(media_type, {})
    model_endpoints_for_type = media_type_config.get("model_endpoints", {})

    if model_name not in model_endpoints_for_type:
        logger.error(
            "Request %s: Model '%s' not configured for media type '%s'.",
            request_id,
            model_name,
            media_type,
        )
        return {"error": "Detection service not available for this media type."}

    model_predict_url = model_endpoints_for_type[model_name]
    logger.info(
        "Request %s: Querying model '%s' (%s) at %s.",
        request_id,
        model_name,
        media_type,
        model_predict_url,
    )

    payload_key = MEDIA_TYPE_PAYLOAD_KEYS.get(media_type)
    if not payload_key:
        logger.error(
            "Request %s: No payload key defined for media type '%s' for model '%s'.",
            request_id,
            media_type,
            model_name,
        )
        return {
            "error": f"Internal configuration error: Payload key not defined for media type '{media_type}'."
        }

    payload = {payload_key: encoded_media_data, "threshold": threshold}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                model_predict_url, json=payload, timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            logger.info(
                "Request %s: Model '%s' (%s) responded successfully (attempt %d).",
                request_id,
                model_name,
                media_type,
                attempt + 1,
            )
            return result
        except requests.exceptions.Timeout:
            logger.warning(
                "Request %s: Timeout querying '%s' (%s) (attempt %d/%d).",
                request_id,
                model_name,
                media_type,
                attempt + 1,
                MAX_RETRIES + 1,
            )
            if attempt == MAX_RETRIES:
                return {
                    "error": "A detection module timed out after multiple attempts."
                }
        except requests.exceptions.HTTPError as e:
            resp = e.response
            status_code = resp.status_code if resp is not None else 0
            error_text = (
                (resp.text[:200] or "(empty body)")
                if resp is not None
                else "No response object."
            )
            logger.error(
                "Request %s: HTTPError from '%s' (%s): %s - %s (attempt %d).",
                request_id,
                model_name,
                media_type,
                status_code,
                error_text,
                attempt + 1,
            )
            if attempt < MAX_RETRIES and status_code in [429, 502, 503, 504]:
                pass
            else:
                return {
                    "error": "A detection module returned an error.",
                    "status_code": status_code,
                }
        except requests.exceptions.RequestException as e:
            logger.error(
                "Request %s: Network error querying '%s' (%s): %s (attempt %d).",
                request_id,
                model_name,
                media_type,
                str(e),
                attempt + 1,
            )
            if attempt == MAX_RETRIES:
                return {"error": "A detection module is temporarily unreachable."}
        except json.JSONDecodeError:
            logger.error(
                "Request %s: Model '%s' (%s) returned non-JSON response: %s (attempt %d).",
                request_id,
                model_name,
                media_type,
                response.text[:100],
                attempt + 1,
            )
            if attempt == MAX_RETRIES:
                return {"error": "A detection module returned an invalid response."}

        if attempt < MAX_RETRIES:
            retry_delay = 2**attempt
            logger.info(
                "Request %s: Retrying '%s' (%s) in %ds...",
                request_id,
                model_name,
                media_type,
                retry_delay,
            )
            time.sleep(retry_delay)

    return {"error": "A detection module failed after multiple retry attempts."}


# ---------------------------------------------------------------------------
# Internal-name anonymization
# ---------------------------------------------------------------------------


def _anonymize_model_keys(results: Dict[str, Any]) -> Dict[str, Any]:
    """Replace internal model-name keys with opaque ``module_N`` labels.

    This prevents accidental leakage of model names if the
    ``result_payload`` (stored in Redis / ``full_response``) is ever
    returned to a user by a new or misconfigured endpoint.

    Args:
        results: Dict keyed by internal model service names.

    Returns:
        New dict with keys replaced by ``module_1``, ``module_2``, etc.
    """
    return {f"module_{i + 1}": v for i, (_, v) in enumerate(sorted(results.items()))}


# ---------------------------------------------------------------------------
# Shared helpers for DB persistence
# ---------------------------------------------------------------------------


def _insert_model_performance(
    db,
    analysis_id: int,
    user_id: Optional[str],
    model_results: dict,
    media_type: str,
) -> None:
    """Insert one ModelPerformance row per model result.

    Args:
        db: SQLAlchemy session.
        analysis_id: FK to analysis_history.id.
        user_id: Supabase UUID of the authenticated user, or None.
        model_results: Dict mapping model name to result dict.
        media_type: "image" | "video" | "audio".
    """
    for model_name, result in model_results.items():
        has_error = "error" in result
        db.add(
            ModelPerformance(
                analysis_id=analysis_id,
                user_id=user_id,
                model_name=model_name,
                model_score=result.get("probability") if not has_error else None,
                raw_output=json.dumps(result) if result else None,
                inference_time_ms=result.get("time_ms"),
                status="error" if has_error else "success",
                error_message=result.get("error") if has_error else None,
                media_type=media_type,
            )
        )


def _insert_cost_tracking(
    db,
    analysis_id: int,
    user_id: Optional[str],
    model_count: int,
    total_inference_ms: Optional[float],
    media_type: str,
    file_size_bytes: Optional[int] = None,
) -> None:
    """Insert one CostTracking row per analysis.

    Args:
        db: SQLAlchemy session.
        analysis_id: FK to analysis_history.id.
        user_id: Supabase UUID of the authenticated user, or None.
        model_count: Number of models that contributed results.
        total_inference_ms: Aggregate inference time in milliseconds.
        media_type: "image" | "video" | "audio".
        file_size_bytes: Size of the uploaded file in bytes, or None.
    """
    db.add(
        CostTracking(
            analysis_id=analysis_id,
            user_id=user_id,
            credits_consumed=1,
            model_count=model_count,
            total_inference_ms=total_inference_ms,
            media_type=media_type,
            file_size_bytes=file_size_bytes,
        )
    )


def _persist_analysis(
    user_id: Optional[str],
    request_id: str,
    media_type: str,
    verdict: str,
    confidence: float,
    ensemble_method: str,
    ensemble_score: float,
    all_results: Dict[str, Dict],
    response_time_ms: Optional[float] = None,
    full_response_json: Optional[str] = None,
    db=None,
) -> None:
    """Persist analysis history, per-model performance, and cost tracking.

    Best-effort: logs warnings on failure, never raises.

    Args:
        user_id: Supabase UUID or None.
        request_id: Detection or job ID.
        media_type: "image" | "video" | "audio".
        verdict: "fake" or "real".
        confidence: Confidence score [0, 1].
        ensemble_method: Ensemble method used.
        ensemble_score: Raw ensemble probability.
        all_results: Dict of model_name -> result dict.
        response_time_ms: Total inference time in ms.
        full_response_json: JSON string of full response payload.
        db: Optional existing DB session (sync path). If None, creates one.
    """

    def _do_persist(session):
        """Inner persistence logic."""
        history_record = AnalysisHistory(
            request_id=request_id,
            user_id=user_id,
            media_type=media_type,
            media_name=None,
            verdict=verdict,
            confidence=confidence,
            ensemble_method=ensemble_method,
            ensemble_score=ensemble_score,
            inference_time=response_time_ms,
            full_response=full_response_json
            or json.dumps({"verdict": verdict, "confidence": confidence}),
        )
        session.add(history_record)
        session.flush()

        num_ok = len([r for r in all_results.values() if "error" not in r])
        _insert_model_performance(
            session, history_record.id, user_id, all_results, media_type
        )
        _insert_cost_tracking(
            session,
            analysis_id=history_record.id,
            user_id=user_id,
            model_count=num_ok,
            total_inference_ms=response_time_ms,
            media_type=media_type,
        )
        session.commit()

    try:
        if db is not None:
            # Sync path: use the caller-provided session.
            _do_persist(db)
        else:
            # Async path: create and manage our own session.
            with get_db_session() as session:
                _do_persist(session)
    except Exception as db_err:
        logger.warning("Detection %s: DB save failed: %s", request_id, db_err)


def _increment_user_scan_usage(user_id: Optional[str], request_id: str) -> None:
    """Increment scan usage for a user. Best-effort, never raises.

    Args:
        user_id: Supabase UUID of the user.
        request_id: For logging only.
    """
    if not user_id:
        return
    try:
        from api_keys import ApiKey

        with get_db_session() as db:
            user_key = (
                db.query(ApiKey)
                .filter(ApiKey.user_id == user_id, ApiKey.revoked_at.is_(None))
                .first()
            )
            if user_key:
                db.query(ApiKey).filter(ApiKey.id == user_key.id).update(
                    {
                        "scans_used": ApiKey.scans_used + 1,
                        "last_used_at": datetime.now(timezone.utc),
                    }
                )
                db.commit()
    except Exception as usage_err:
        logger.warning(
            "Detection %s: Usage increment failed: %s", request_id, usage_err
        )


def _get_max_provenance_score(results: Dict[str, Dict]) -> float:
    """Return the highest provenance probability from results.

    Args:
        results: Dict of model_name -> result dict.

    Returns:
        Max provenance score, or 0.0 if none available.
    """
    max_score = 0.0
    for name, result in results.items():
        if isinstance(result, dict) and "error" not in result:
            prob = result.get("probability", 0.5)
            if prob > max_score:
                max_score = prob
    return max_score


def _fan_out_to_services(
    service_names: List[str],
    media_type: str,
    encoded_media: str,
    threshold: float,
    request_id: str,
    max_workers: int = 12,
    redis_update_lock=None,
    redis_key: Optional[str] = None,
) -> Dict[str, Dict]:
    """Fan out to model/provenance services in parallel.

    Args:
        service_names: List of model/service names to query.
        media_type: Media category.
        encoded_media: Base64-encoded media bytes.
        threshold: Fake probability threshold.
        request_id: Request/job ID for logging.
        max_workers: Max concurrent threads.
        redis_update_lock: Optional lock for async Redis updates.
        redis_key: Optional Redis key for per-model status updates.

    Returns:
        Dict of model_name -> result dict.
    """
    results: Dict[str, Dict] = {}
    # Map internal model names to opaque labels for Redis progress.
    _anon_map = {
        name: f"module_{i + 1}" for i, name in enumerate(sorted(service_names))
    }

    def _call(m_name: str):
        return m_name, query_model_api(
            m_name, media_type, encoded_media, threshold, request_id
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(service_names), max_workers)
    ) as executor:
        futures = {executor.submit(_call, m): m for m in service_names}
        for future in concurrent.futures.as_completed(futures):
            m_name = futures[future]
            try:
                _, result = future.result()
                results[m_name] = result
                if "error" in result:
                    model_entry = {"status": "error", "error": result["error"]}
                else:
                    prob = result.get("probability", 0.0)
                    model_entry = {
                        "status": "complete",
                        "probability": prob,
                        "is_fake": prob >= threshold,
                    }
            except Exception as exc:
                results[m_name] = {"error": str(exc)}
                model_entry = {"status": "error", "error": str(exc)}
                analytics.track(
                    None,
                    "model_timeout",
                    {
                        "model_name": m_name,
                        "media_type": media_type,
                        "timeout_ms": 0,
                    },
                )

            # Write per-model status to Redis for async job progress.
            # Uses anonymized key to prevent model-name leakage.
            if redis_update_lock is not None and redis_key:
                anon_key = _anon_map.get(m_name, m_name)
                with redis_update_lock:
                    raw = redis_client.get(redis_key)
                    state = json.loads(raw) if raw else {}
                    state.setdefault("models", {})[anon_key] = model_entry
                    redis_client.set(redis_key, json.dumps(state))
                logger.info(
                    "Job %s: Service '%s' -> %s",
                    request_id,
                    m_name,
                    model_entry["status"],
                )

    return results


def _resolve_endpoints(media_type: str, models_list: Optional[List[str]] = None):
    """Resolve model and provenance endpoints for a media type.

    Args:
        media_type: "image" | "video" | "audio".
        models_list: Optional list of model names to filter to.

    Returns:
        Tuple of (provenance_names, model_names).
    """
    media_type_config = (
        ALL_MODEL_CONFIGS.get("media_types", {}) if ALL_MODEL_CONFIGS else {}
    ).get(media_type, {})
    all_endpoints = media_type_config.get("model_endpoints", {})

    if models_list:
        all_endpoints = {k: v for k, v in all_endpoints.items() if k in models_list}

    provenance_names = [k for k in all_endpoints if k in PROVENANCE_SERVICES]
    model_names = [k for k in all_endpoints if k not in PROVENANCE_SERVICES]
    return provenance_names, model_names


# ---------------------------------------------------------------------------
# Public API: Sync detection
# ---------------------------------------------------------------------------


def _run_sync_detection(
    file_bytes: bytes,
    content_type: str,
    media_type: str,
    user_id: Optional[str],
    detection_id: str,
    db,
    threshold_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Run synchronous detection for the public v1 API.

    Fans out to all configured model microservices in parallel, computes the
    ensemble verdict, and persists the result.

    Args:
        file_bytes: Raw bytes of the uploaded media file.
        content_type: MIME type of the file (e.g. 'image/jpeg').
        media_type: Resolved media category ('image', 'video', 'audio').
        user_id: API key owner's user ID (for analytics).
        detection_id: The det_XXXXXXXX ID assigned to this request.
        db: SQLAlchemy session (used to persist analysis history).
        threshold_override: Optional threshold override (0.0–1.0).

    Returns:
        Dict with 'verdict' (str), 'confidence' (float), and 'media_type' (str).

    Raises:
        HTTPException: 503 if all models fail.
    """
    threshold = float(
        ALL_MODEL_CONFIGS.get("default_threshold", 0.5) if ALL_MODEL_CONFIGS else 0.5
    )
    if threshold_override is not None:
        threshold = threshold_override
    ensemble_method = str(
        ALL_MODEL_CONFIGS.get("default_ensemble_method", "average")
        if ALL_MODEL_CONFIGS
        else "average"
    )

    encoded_media = base64.b64encode(file_bytes).decode("utf-8")

    provenance_names, model_names = _resolve_endpoints(media_type)

    # Single-phase parallel dispatch: all models + provenance at once.
    all_service_names = list(model_names) + list(provenance_names)
    if not all_service_names:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "no_models",
                "message": f"No models configured for '{media_type}'.",
            },
        )

    all_results = _fan_out_to_services(
        all_service_names,
        media_type,
        encoded_media,
        threshold,
        detection_id,
        max_workers=max(len(all_service_names), 12),
    )

    # Provenance override still checked, just after parallel dispatch.
    max_prov = _get_max_provenance_score(all_results)
    if max_prov >= PROVENANCE_OVERRIDE_THRESHOLD:
        logger.info(
            "Detection %s: Provenance override triggered (score=%.4f >= %.2f).",
            detection_id,
            max_prov,
            PROVENANCE_OVERRIDE_THRESHOLD,
        )
        _persist_analysis(
            user_id,
            detection_id,
            media_type,
            "fake",
            max_prov,
            "provenance_override",
            max_prov,
            all_results,
            db=db,
        )
        return {"verdict": "fake", "confidence": max_prov, "media_type": media_type}

    if not model_names:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "no_models",
                "message": f"No detection models configured for '{media_type}'.",
            },
        )

    # Check at least one model succeeded.
    model_ok = any(
        "error" not in r
        for name, r in all_results.items()
        if name not in PROVENANCE_SERVICES
    )
    if not model_ok:
        # If every model returned a client-error (400), the input is likely
        # invalid rather than the service being unavailable.
        detection_errors = {
            name: r
            for name, r in all_results.items()
            if name not in PROVENANCE_SERVICES and "error" in r
        }
        all_client_errors = all(
            r.get("status_code") == 400 for r in detection_errors.values()
        )
        if all_client_errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_media",
                    "message": (
                        "The uploaded file could not be processed by any "
                        "detection model. Please verify it is a valid, "
                        "non-corrupted media file."
                    ),
                },
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "all_models_failed",
                "message": "All detection models failed. Please try again.",
            },
        )

    (verdict, confidence, fake_votes, real_votes, ensemble_prob_fake, actual_method) = (
        calculate_ensemble_verdict_api(
            all_results, threshold, ensemble_method, media_type, detection_id
        )
    )

    _persist_analysis(
        user_id,
        detection_id,
        media_type,
        verdict,
        confidence,
        actual_method,
        ensemble_prob_fake,
        all_results,
        db=db,
    )

    return {"verdict": verdict, "confidence": confidence, "media_type": media_type}


# ---------------------------------------------------------------------------
# Public API: Async detection (Celery)
# ---------------------------------------------------------------------------


def _enqueue_async_detection(
    file_bytes: bytes,
    content_type: str,
    media_type: str,
    user_id: Optional[str],
    detection_id: str,
) -> None:
    """Upload media to MinIO and enqueue a Celery detection task.

    Writes QUEUED state to Redis so the caller can immediately poll
    /v1/results/{detection_id}.

    Args:
        file_bytes: Raw bytes of the uploaded media file.
        content_type: MIME type of the file (e.g. 'image/jpeg').
        media_type: Resolved media category ('image', 'video', 'audio').
        user_id: API key owner's user ID.
        detection_id: The det_XXXXXXXX ID assigned to this request (used as job_id).

    Raises:
        HTTPException: 500 if MinIO upload or Celery enqueue fails.
    """
    # Import run_detection lazily to avoid circular imports at module load time.
    from main import run_detection

    ext = CONTENT_TYPE_TO_EXT.get(content_type, "bin")
    threshold = float(
        ALL_MODEL_CONFIGS.get("default_threshold", 0.5) if ALL_MODEL_CONFIGS else 0.5
    )
    ensemble_method = str(
        ALL_MODEL_CONFIGS.get("default_ensemble_method", "average")
        if ALL_MODEL_CONFIGS
        else "average"
    )

    if minio_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "async_unavailable",
                "message": (
                    "Async detection is temporarily unavailable "
                    "(object storage not configured). "
                    "Please use synchronous mode."
                ),
            },
        )

    redis_key = f"job:{detection_id}"
    job_state = {
        "job_id": detection_id,
        "user_id": user_id,
        "status": "QUEUED",
        "media_type": media_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
        "result": None,
        "error": None,
    }
    redis_client.set(redis_key, json.dumps(job_state), ex=7200)

    object_path = f"{detection_id}/media.{ext}"
    try:
        minio_client.put_object(
            MINIO_JOBS_BUCKET,
            object_path,
            io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type,
        )
    except Exception as minio_err:
        redis_client.delete(redis_key)
        logger.error("Detection %s: MinIO upload failed: %s", detection_id, minio_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "storage_error",
                "message": "Failed to store uploaded file. Please try again.",
            },
        ) from minio_err

    try:
        run_detection.delay(
            detection_id,
            media_type,
            ext,
            threshold,
            ensemble_method,
            None,  # models_list — use all configured
            user_id,
        )
    except Exception as celery_err:
        redis_client.delete(redis_key)
        try:
            minio_client.remove_object(MINIO_JOBS_BUCKET, object_path)
        except Exception:
            pass
        logger.error(
            "Detection %s: Celery enqueue failed: %s", detection_id, celery_err
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "queue_error",
                "message": "Failed to enqueue detection job. Please try again.",
            },
        ) from celery_err


# ---------------------------------------------------------------------------
# Celery task body
# ---------------------------------------------------------------------------


def run_detection_task(
    redis_update_lock: object,
    job_id: str,
    media_type: str,
    ext: str,
    threshold: float,
    ensemble_method: str,
    models_list: Optional[List[str]],
    user_id: Optional[str] = None,
) -> None:
    """Execute the Celery detection job body.

    Separated from the ``@celery_app.task`` decorator in ``main.py`` to keep
    ``main.py`` concise.  The lock argument is the module-level
    ``_redis_model_update_lock`` from ``main.py``.

    Args:
        redis_update_lock: Threading lock used to serialize per-model Redis writes.
        job_id: UUID of the job.
        media_type: "image" | "video" | "audio".
        ext: File extension (e.g. "jpg", "wav", "mp4").
        threshold: Fake probability threshold (0.0-1.0).
        ensemble_method: Ensemble strategy name.
        models_list: List of model names to run, or None for all configured.
        user_id: Supabase UUID of the authenticated user, or None for anonymous.
    """
    redis_key = f"job:{job_id}"

    def _update_redis(patch: dict) -> None:
        raw = redis_client.get(redis_key)
        state = json.loads(raw) if raw else {}
        state.update(patch)
        redis_client.set(redis_key, json.dumps(state))

    def _mark_complete(result_payload: dict) -> None:
        _update_redis(
            {
                "status": "COMPLETE",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "result": result_payload,
                "error": None,
            }
        )
        redis_client.expire(redis_key, 3600)

    try:
        if minio_client is None:
            raise RuntimeError("MinIO client is not configured; cannot run async job.")

        # Step 1: Resolve endpoints.
        provenance_names, model_names = _resolve_endpoints(media_type, models_list)
        all_service_names = provenance_names + model_names

        initial_models = {
            f"module_{i + 1}": {"status": "pending"}
            for i, m in enumerate(sorted(all_service_names))
        }
        _update_redis({"status": "PROCESSING", "models": initial_models})
        redis_client.expire(redis_key, 7200)
        logger.info(
            "Job %s: PROCESSING — %d service(s): %s",
            job_id,
            len(all_service_names),
            all_service_names,
        )

        # Step 2: Download file from MinIO.
        task_start_time = time.time()
        object_path = f"{job_id}/media.{ext}"
        response = minio_client.get_object(MINIO_JOBS_BUCKET, object_path)
        with response:
            file_bytes = response.read()
        logger.info("Job %s: Downloaded %d bytes from MinIO.", job_id, len(file_bytes))

        # Step 3: Base64-encode for model services.
        encoded_media = base64.b64encode(file_bytes).decode("utf-8")

        # Single-phase parallel dispatch: all models + provenance at once.
        all_service_names_dispatch = list(model_names) + list(provenance_names)
        if not all_service_names_dispatch:
            raise RuntimeError(
                f"No services configured for job {job_id} ({media_type})."
            )

        all_results = _fan_out_to_services(
            all_service_names_dispatch,
            media_type,
            encoded_media,
            threshold,
            job_id,
            max_workers=max(len(all_service_names_dispatch), 12),
            redis_update_lock=redis_update_lock,
            redis_key=redis_key,
        )

        # Provenance override still checked, just after parallel dispatch.
        max_prov = _get_max_provenance_score(all_results)
        if max_prov >= PROVENANCE_OVERRIDE_THRESHOLD:
            logger.info(
                "Job %s: Provenance override triggered (score=%.4f >= %.2f).",
                job_id,
                max_prov,
                PROVENANCE_OVERRIDE_THRESHOLD,
            )
            num_ok = len([r for r in all_results.values() if "error" not in r])
            elapsed_ms = round((time.time() - task_start_time) * 1000, 1)
            result_payload = {
                "request_id": job_id,
                "is_likely_deepfake": True,
                "deepfake_probability": max_prov,
                "model_count": num_ok,
                "fake_votes": num_ok,
                "real_votes": 0,
                "response_time": elapsed_ms,
                "ensemble_method_used": "provenance_override",
                "model_results": _anonymize_model_keys(all_results),
                "processing_mode": "CPU-only",
                "media_type_processed": media_type,
            }

            _persist_analysis(
                user_id,
                job_id,
                media_type,
                "fake",
                max_prov,
                "provenance_override",
                max_prov,
                all_results,
                elapsed_ms,
                json.dumps(result_payload),
            )
            _mark_complete(result_payload)
            _increment_user_scan_usage(user_id, job_id)
            logger.info("Job %s: COMPLETE (provenance override).", job_id)
            return

        if not model_names:
            raise RuntimeError(
                f"No detection models configured for job {job_id} ({media_type})."
            )

        # Step 5: Check at least one model succeeded.
        model_ok = any(
            "error" not in r
            for name, r in all_results.items()
            if name not in PROVENANCE_SERVICES
        )
        if not model_ok:
            raise RuntimeError(f"All models failed for job {job_id} ({media_type}).")

        # Step 6: Compute ensemble verdict.
        (
            verdict,
            confidence,
            fake_votes,
            real_votes,
            ensemble_prob_fake,
            actual_method,
        ) = calculate_ensemble_verdict_api(
            all_results, threshold, ensemble_method, media_type, job_id
        )

        num_models_ok = len([r for r in all_results.values() if "error" not in r])
        elapsed_ms = round((time.time() - task_start_time) * 1000, 1)
        result_payload = {
            "request_id": job_id,
            "is_likely_deepfake": verdict == "fake",
            "deepfake_probability": ensemble_prob_fake,
            "model_count": num_models_ok,
            "fake_votes": fake_votes,
            "real_votes": real_votes,
            "response_time": elapsed_ms,
            "ensemble_method_used": actual_method,
            "model_results": _anonymize_model_keys(all_results),
            "processing_mode": "CPU-only",
            "media_type_processed": media_type,
        }

        # Step 7: Persist and complete.
        _persist_analysis(
            user_id,
            job_id,
            media_type,
            verdict,
            confidence,
            actual_method,
            ensemble_prob_fake,
            all_results,
            elapsed_ms,
            json.dumps(result_payload),
        )
        analytics.track(
            user_id,
            "analysis_completed",
            {
                "job_id": job_id,
                "media_type": media_type,
                "verdict": verdict,
                "confidence": confidence,
                "ensemble_method": actual_method,
                "total_inference_ms": elapsed_ms,
                "model_count": num_models_ok,
            },
        )
        _mark_complete(result_payload)
        _increment_user_scan_usage(user_id, job_id)
        logger.info("Job %s: COMPLETE.", job_id)

    except Exception as exc:
        logger.exception("Job %s: Task failed: %s", job_id, exc)
        analytics.track(
            user_id,
            "analysis_failed",
            {
                "job_id": job_id,
                "media_type": media_type,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            },
        )
        try:
            _update_redis(
                {
                    "status": "FAILED",
                    "error": "Analysis failed. Please try again or contact support.",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            redis_client.expire(redis_key, 3600)
        except Exception as redis_err:
            logger.error(
                "Job %s: Could not write FAILED to Redis: %s", job_id, redis_err
            )
    finally:
        # Step 9: Clean up MinIO (best-effort).
        try:
            minio_client.remove_object(MINIO_JOBS_BUCKET, f"{job_id}/media.{ext}")
            logger.info("Job %s: MinIO object deleted.", job_id)
        except Exception as minio_err:
            logger.warning("Job %s: MinIO cleanup failed: %s", job_id, minio_err)
