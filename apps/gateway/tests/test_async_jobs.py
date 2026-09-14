"""
Tests for async job queue pipeline:
- celery_app module
- run_detection Celery task
- /detect endpoint (202 + job_id)
- GET /jobs/{job_id} endpoint
- Security: BOLA protection, credit drain prevention, error sanitization
"""

import json
from unittest.mock import MagicMock, patch

import pytest

TEST_USER_ID = "test-user-abc-123"
OTHER_USER_ID = "other-user-xyz-789"


# --- Celery eager mode: run tasks synchronously in tests (no broker needed) ---
# Must be set before importing main or celery_app.
@pytest.fixture(autouse=True)
def celery_eager():
    """Run Celery tasks synchronously in tests (no broker required)."""
    import main  # noqa: PLC0415

    main.celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
    yield
    main.celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)


@pytest.fixture(autouse=True)
def mock_auth():
    """Override auth dependency to return a valid Supabase JWT payload."""
    from auth import get_current_user
    from main import app

    async def _mock_user():
        return {
            "sub": TEST_USER_ID,
            "email": "test@example.com",
            "aud": "authenticated",
        }

    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_current_user] = _mock_user
    yield
    app.dependency_overrides = previous_overrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_redis():
    """Returns a mock Redis client with a backing dict for get/set/delete/expire."""
    store = {}
    r = MagicMock()
    r.set.side_effect = lambda key, val, ex=None: store.update({key: val})
    r.get.side_effect = lambda key: store.get(key)
    r.delete.side_effect = lambda key: store.pop(key, None)
    r.expire.side_effect = lambda key, ttl: None
    return r, store


# ---------------------------------------------------------------------------
# celery_app module
# ---------------------------------------------------------------------------


def test_celery_app_importable():
    """celery_app module must export a Celery instance named celery_app."""
    from celery import Celery
    from celery_app import celery_app

    assert isinstance(celery_app, Celery)


def test_celery_app_uses_redis_broker():
    """Broker URL must point to Redis (default or env-configured)."""
    from celery_app import celery_app

    assert celery_app.conf.broker_url.startswith("redis://")


def test_celery_app_includes_main():
    """Worker must be told to import main.py to discover tasks."""
    from celery_app import celery_app

    assert "main" in (celery_app.conf.include or [])


# ---------------------------------------------------------------------------
# run_detection Celery task
# ---------------------------------------------------------------------------


def test_run_detection_task_sets_processing_then_complete():
    """
    run_detection task must:
    - Set job status to PROCESSING in Redis
    - Call model services
    - Set job status to COMPLETE with a result
    """
    import io

    import main
    from PIL import Image as PILImage

    fake_redis, store = _make_fake_redis()

    # Seed QUEUED state (as /detect would have written)
    initial_state = {
        "job_id": "test-job-1",
        "user_id": TEST_USER_ID,
        "status": "QUEUED",
        "media_type": "image",
        "created_at": "2026-03-20T00:00:00",
        "models": {},
        "result": None,
        "error": None,
    }
    store["job:test-job-1"] = json.dumps(initial_state)

    fake_minio = MagicMock()
    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(255, 255, 255)).save(buf, format="JPEG")
    image_bytes = buf.getvalue()
    fake_minio.get_object.return_value.__enter__ = MagicMock(
        return_value=fake_minio.get_object.return_value
    )
    fake_minio.get_object.return_value.__exit__ = MagicMock(return_value=False)
    fake_minio.get_object.return_value.read.return_value = image_bytes
    fake_minio.remove_object = MagicMock()

    # prediction must be integer 0 (real) or 1 (fake) — matches calculate_ensemble_verdict_api's voting logic
    model_result = {"probability": 0.2, "prediction": 0, "is_fake": False}

    import services.detection as _detection_module

    # Provide a config with the efficientnet model so _resolve_endpoints finds it.
    fake_config = {
        "media_types": {
            "image": {
                "model_endpoints": {
                    "efficientnet": "http://fake:5001/predict",
                },
            },
        },
        "default_threshold": 0.5,
    }

    with (
        patch.object(_detection_module, "redis_client", fake_redis),
        patch.object(_detection_module, "minio_client", fake_minio),
        patch.object(_detection_module, "query_model_api", return_value=model_result),
        patch.object(_detection_module, "ALL_MODEL_CONFIGS", fake_config),
    ):
        main.run_detection.apply(
            args=["test-job-1", "image", "jpg", 0.5, "average", ["efficientnet"]]
        )

    final_raw = store.get("job:test-job-1")
    assert final_raw is not None, "Redis key must still exist after completion"
    final = json.loads(final_raw)
    assert final["status"] == "COMPLETE"
    assert final["result"] is not None
    assert final["result"]["is_likely_deepfake"] is False


def test_run_detection_task_sets_failed_on_error():
    """If MinIO download raises, task must write FAILED status to Redis with a generic error."""
    import main

    fake_redis, store = _make_fake_redis()
    store["job:test-job-2"] = json.dumps(
        {
            "job_id": "test-job-2",
            "user_id": TEST_USER_ID,
            "status": "QUEUED",
            "media_type": "audio",
            "created_at": "2026-03-20T00:00:00",
            "models": {},
            "result": None,
            "error": None,
        }
    )

    fake_minio = MagicMock()
    fake_minio.get_object.side_effect = Exception("MinIO connection refused")

    import services.detection as _detection_module

    with (
        patch.object(_detection_module, "redis_client", fake_redis),
        patch.object(_detection_module, "minio_client", fake_minio),
    ):
        main.run_detection.apply(
            args=["test-job-2", "audio", "wav", 0.5, "average", None]
        )

    final = json.loads(store["job:test-job-2"])
    assert final["status"] == "FAILED"
    # Error message must be generic — no internal details leaked to user
    assert "MinIO" not in final["error"]
    assert "try again" in final["error"].lower() or "support" in final["error"].lower()


# ---------------------------------------------------------------------------
# /detect endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    import main
    from fastapi.testclient import TestClient

    return TestClient(main.app)


@pytest.mark.skip(
    reason="Old /detect endpoint removed — async detection tested via /v1/detect in test_v1_router.py"
)
def test_detect_returns_202_with_job_id(client):
    """POST /detect must return 202 immediately with a job_id, not block."""
    import io

    import main
    from PIL import Image as PILImage

    fake_redis, store = _make_fake_redis()
    fake_minio = MagicMock()
    fake_minio.put_object = MagicMock()

    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(255, 255, 255)).save(buf, format="JPEG")
    buf.seek(0)

    with (
        patch.object(main, "redis_client", fake_redis),
        patch.object(main, "minio_client", fake_minio),
        patch("main.run_detection.delay", return_value=MagicMock()),
        patch("main.deduct_credit", return_value=True),
    ):
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "QUEUED"
    assert data["media_type"] == "image"


@pytest.mark.skip(reason="Old /detect endpoint removed")
def test_detect_stores_user_id_in_redis(client):
    """POST /detect must write user_id into the Redis job state."""
    import io

    import main
    from PIL import Image as PILImage

    fake_redis, store = _make_fake_redis()
    fake_minio = MagicMock()
    fake_minio.put_object = MagicMock()

    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(255, 255, 255)).save(buf, format="JPEG")
    buf.seek(0)

    with (
        patch.object(main, "redis_client", fake_redis),
        patch.object(main, "minio_client", fake_minio),
        patch("main.run_detection.delay", return_value=MagicMock()),
        patch("main.deduct_credit", return_value=True),
    ):
        response = client.post(
            "/detect",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    redis_state = json.loads(store[f"job:{job_id}"])
    assert redis_state["user_id"] == TEST_USER_ID


@pytest.mark.skip(reason="Old /detect endpoint removed")
def test_detect_returns_415_for_unsupported_type(client):
    """POST /detect must reject unsupported content types before deducting credits."""
    import main

    fake_redis, _ = _make_fake_redis()
    fake_minio = MagicMock()
    mock_deduct = MagicMock(return_value=True)

    with (
        patch.object(main, "redis_client", fake_redis),
        patch.object(main, "minio_client", fake_minio),
        patch("main.deduct_credit", mock_deduct),
    ):
        response = client.post(
            "/detect",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 415
    # Credit must NOT have been deducted for an unsupported file type
    mock_deduct.assert_not_called()


# ---------------------------------------------------------------------------
# GET /jobs/{job_id} endpoint
# ---------------------------------------------------------------------------


def test_get_job_returns_processing_state(client):
    """GET /jobs/{job_id} must return 200 with live Redis state for the job owner."""
    import main

    fake_redis, store = _make_fake_redis()
    state = {
        "job_id": "abc-123",
        "user_id": TEST_USER_ID,
        "status": "PROCESSING",
        "media_type": "audio",
        "created_at": "2026-03-20T00:00:00",
        "completed_at": None,
        "models": {"aasist": {"status": "complete", "probability": 0.91}},
        "result": None,
        "error": None,
    }
    store["job:abc-123"] = json.dumps(state)

    import routers.history as _history_module

    with patch.object(_history_module, "redis_client", fake_redis):
        response = client.get("/jobs/abc-123")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PROCESSING"
    assert data["media_type"] == "audio"
    # Internal model names must NOT be exposed to clients.
    assert "models" not in data
    assert "user_id" not in data


def test_get_job_returns_404_for_unknown(client):
    """GET /jobs/{job_id} returns 404 if not in Redis or database."""
    import main

    fake_redis, _ = _make_fake_redis()  # empty store

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.filter.return_value.first.return_value = (
        None
    )

    import routers.history as _history_module

    with (
        patch.object(_history_module, "redis_client", fake_redis),
        patch(
            "routers.history.get_db_session",
            return_value=MagicMock(
                __enter__=MagicMock(return_value=mock_db),
                __exit__=MagicMock(return_value=False),
            ),
        ),
    ):
        response = client.get("/jobs/nonexistent-id")

    assert response.status_code == 404


def test_get_job_returns_404_for_other_users_job(client):
    """GET /jobs/{job_id} must return 404 when accessing another user's job (BOLA protection)."""
    import main

    fake_redis, store = _make_fake_redis()
    state = {
        "job_id": "other-job-456",
        "user_id": OTHER_USER_ID,
        "status": "COMPLETE",
        "media_type": "image",
        "created_at": "2026-03-20T00:00:00",
        "completed_at": "2026-03-20T00:01:00",
        "models": {},
        "result": {"is_likely_deepfake": True},
        "error": None,
    }
    store["job:other-job-456"] = json.dumps(state)

    import routers.history as _history_module

    with patch.object(_history_module, "redis_client", fake_redis):
        response = client.get("/jobs/other-job-456")

    assert response.status_code == 404


def test_get_job_falls_back_to_database(client):
    """GET /jobs/{job_id} returns synthetic COMPLETE from database if Redis key expired."""
    import main

    fake_redis, _ = _make_fake_redis()  # empty — simulates expired TTL

    result_payload = {"is_likely_deepfake": True, "deepfake_probability": 0.97}
    db_record = MagicMock()
    db_record.media_type = "video"
    db_record.timestamp = MagicMock()
    db_record.timestamp.isoformat.return_value = "2026-03-20T10:00:00"
    db_record.full_response = json.dumps(result_payload)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.filter.return_value.first.return_value = (
        db_record
    )

    import routers.history as _history_module

    with (
        patch.object(_history_module, "redis_client", fake_redis),
        patch(
            "routers.history.get_db_session",
            return_value=MagicMock(
                __enter__=MagicMock(return_value=mock_db),
                __exit__=MagicMock(return_value=False),
            ),
        ),
    ):
        response = client.get("/jobs/expired-job-id")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETE"
    assert data["media_type"] == "video"
    assert data["result"]["is_likely_deepfake"] is True
    # The DB-fallback response does not include a "models" key —
    # only job_id, status, media_type, created_at, completed_at,
    # result, and error.
    assert "models" not in data
