"""Comprehensive tests for the POST /v1/detect endpoint.

Covers: auth, file type validation, quota, rate limiting, sync/async modes,
response schema, and file size limits.
"""

import io
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_jpeg_bytes() -> bytes:
    """Generate a minimal valid 1x1 JPEG that PIL can verify."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 0)).save(buf, "JPEG")
    return buf.getvalue()


def _make_png_bytes() -> bytes:
    """Generate a minimal valid 1x1 PNG that PIL can verify."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


_MINIMAL_JPEG = _make_jpeg_bytes()
_MINIMAL_PNG = _make_png_bytes()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_api_key(
    tier: str = "free",
    scans_used: int = 0,
    user_id: str = "user-test-123",
    jwt_auth: bool = False,
):
    """Return a minimal mock ApiKey ORM object."""
    key = MagicMock()
    key.tier = tier
    key.scans_used = scans_used
    key.key_hash = "testhash"
    key.user_id = user_id
    key.id = "key-id-001"
    key.period_start = datetime(2026, 3, 1)
    key.revoked_at = None
    return key


def _make_auth_result(api_key=None, is_jwt=False):
    """Wrap an api_key in an AuthResult for mocking _authenticate."""
    if api_key is None:
        api_key = _make_api_key()
    result = MagicMock()
    result.api_key = api_key
    result.is_jwt_auth = is_jwt
    return result


def _fake_result(
    verdict: str = "fake", confidence: float = 0.91, media_type: str = "image"
):
    return {"verdict": verdict, "confidence": confidence, "media_type": media_type}


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_client():
    """Return a TestClient with Redis / MinIO patched at import time."""
    with patch("redis.from_url"), patch("minio.Minio"):
        from main import app

        client = TestClient(app, raise_server_exceptions=False)
        yield client


# ---------------------------------------------------------------------------
# 1. Valid image upload — 200 with verdict, confidence, media_type
# ---------------------------------------------------------------------------


class TestValidImageUpload:
    """Valid JPEG upload should return 200 with required fields."""

    def test_returns_200(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection",
                return_value=_fake_result("fake", 0.91, "image"),
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={
                    "file": (
                        "photo.jpg",
                        _MINIMAL_JPEG,
                        "image/jpeg",
                    )
                },
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 200

    def test_response_has_verdict(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection",
                return_value=_fake_result("fake", 0.91, "image"),
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={
                    "file": (
                        "photo.jpg",
                        _MINIMAL_JPEG,
                        "image/jpeg",
                    )
                },
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        assert body["verdict"] == "fake"

    def test_response_has_confidence(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection",
                return_value=_fake_result("fake", 0.91, "image"),
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={
                    "file": (
                        "photo.jpg",
                        _MINIMAL_JPEG,
                        "image/jpeg",
                    )
                },
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        assert body["confidence"] == 0.91

    def test_response_has_media_type(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection",
                return_value=_fake_result("fake", 0.91, "image"),
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={
                    "file": (
                        "photo.jpg",
                        _MINIMAL_JPEG,
                        "image/jpeg",
                    )
                },
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        assert body["media_type"] == "image"

    def test_response_id_has_det_prefix(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection", return_value=_fake_result()
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={
                    "file": (
                        "photo.jpg",
                        _MINIMAL_JPEG,
                        "image/jpeg",
                    )
                },
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        assert body["id"].startswith("det_")

    def test_response_has_created_at(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection", return_value=_fake_result()
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={
                    "file": (
                        "photo.jpg",
                        _MINIMAL_JPEG,
                        "image/jpeg",
                    )
                },
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        assert "created_at" in body


# ---------------------------------------------------------------------------
# 2. Valid audio upload — 200
# ---------------------------------------------------------------------------


class TestValidAudioUpload:
    """Valid WAV upload should return 200."""

    def test_wav_returns_200(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection",
                return_value=_fake_result("real", 0.78, "audio"),
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("clip.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["media_type"] == "audio"

    def test_mp3_returns_200(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection",
                return_value=_fake_result("real", 0.65, "audio"),
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("clip.mp3", b"\xff\xfb" + b"\x00" * 100, "audio/mpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 3. Missing auth — 401
# ---------------------------------------------------------------------------


class TestMissingAuth:
    """With auth required, requests without a valid header must return 401.

    Auth is optional by default so a local install works unconfigured, so these
    tests pin REQUIRE_AUTH on explicitly. TestAuthDisabled below covers the
    other mode.
    """

    @pytest.fixture(autouse=True)
    def _require_auth(self, monkeypatch):
        import routers.v1 as v1

        monkeypatch.setattr(v1, "REQUIRE_AUTH", True)

    def test_no_header_returns_401(self, test_client):
        response = test_client.post(
            "/v1/detect",
            files={"file": ("photo.jpg", b"data", "image/jpeg")},
        )
        assert response.status_code == 401

    def test_wrong_scheme_returns_401(self, test_client):
        response = test_client.post(
            "/v1/detect",
            files={"file": ("photo.jpg", b"data", "image/jpeg")},
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert response.status_code == 401

    def test_empty_bearer_returns_401(self, test_client):
        response = test_client.post(
            "/v1/detect",
            files={"file": ("photo.jpg", b"data", "image/jpeg")},
            headers={"Authorization": "Bearer "},
        )
        assert response.status_code == 401

    def test_401_error_code_is_unauthorized(self, test_client):
        response = test_client.post(
            "/v1/detect",
            files={"file": ("photo.jpg", b"data", "image/jpeg")},
        )
        body = response.json()
        detail = body.get("detail") or body
        assert detail.get("error") == "unauthorized"


# ---------------------------------------------------------------------------
# 4. Invalid / revoked API key — 401
# ---------------------------------------------------------------------------


class TestInvalidApiKey:
    """Invalid or revoked API keys must return 401."""

    def test_invalid_key_returns_401(self, test_client):
        from fastapi import HTTPException

        def _raise(*args, **kwargs):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Invalid or revoked API key.",
                },
            )

        with patch("routers.v1._authenticate", side_effect=_raise):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("photo.jpg", b"data", "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_badkey"},
            )
        assert response.status_code == 401

    def test_revoked_key_returns_401(self, test_client):
        from fastapi import HTTPException

        def _raise(*args, **kwargs):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Invalid or revoked API key.",
                },
            )

        with patch("routers.v1._authenticate", side_effect=_raise):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("photo.jpg", b"data", "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_revoked"},
            )
        assert response.status_code == 401

    def test_revoked_key_error_code(self, test_client):
        from fastapi import HTTPException

        def _raise(*args, **kwargs):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Invalid or revoked API key.",
                },
            )

        with patch("routers.v1._authenticate", side_effect=_raise):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("photo.jpg", b"data", "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_revoked"},
            )
        body = response.json()
        detail = body.get("detail") or body
        assert detail.get("error") == "unauthorized"


# ---------------------------------------------------------------------------
# 5. Invalid file type — 415
# ---------------------------------------------------------------------------


class TestInvalidFileType:
    """Unsupported MIME types must return 415."""

    def test_text_file_returns_415(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("readme.txt", b"hello world", "text/plain")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 415

    def test_pdf_returns_415(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 415

    def test_415_error_code(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("data.csv", b"a,b,c", "text/csv")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        detail = body.get("detail") or body
        assert detail.get("error") == "unsupported_media_type"


# ---------------------------------------------------------------------------
# 6. Quota exceeded — 402
# ---------------------------------------------------------------------------


class TestQuotaExceeded:
    """When monthly quota is exhausted, the endpoint must return 402."""

    def test_quota_exceeded_returns_402(self, test_client):
        api_key = _make_api_key(tier="free", scans_used=200)

        from fastapi import HTTPException

        def _raise(*args, **kwargs):
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "quota_exceeded",
                    "message": "Lifetime quota of 200 reached.",
                },
            )

        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", side_effect=_raise),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("photo.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 402

    def test_quota_exceeded_error_code(self, test_client):
        api_key = _make_api_key(tier="free", scans_used=200)

        from fastapi import HTTPException

        def _raise(*args, **kwargs):
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "quota_exceeded",
                    "message": "Lifetime quota reached.",
                },
            )

        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", side_effect=_raise),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("photo.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        detail = body.get("detail") or body
        assert detail.get("error") == "quota_exceeded"


# ---------------------------------------------------------------------------
# 7. Rate limited — 429
# ---------------------------------------------------------------------------


class TestRateLimited:
    """When per-minute rate limit is hit, the endpoint must return 429."""

    def test_rate_limited_returns_429(self, test_client):
        api_key = _make_api_key()

        from fastapi import HTTPException

        def _raise(*args, **kwargs):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": "Rate limit exceeded. Retry after 10s.",
                    "retry_after": 10,
                },
            )

        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", side_effect=_raise),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("photo.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 429

    def test_rate_limited_error_code(self, test_client):
        api_key = _make_api_key()

        from fastapi import HTTPException

        def _raise(*args, **kwargs):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": "Too many requests.",
                    "retry_after": 5,
                },
            )

        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", side_effect=_raise),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("photo.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        detail = body.get("detail") or body
        assert detail.get("error") == "rate_limited"


# ---------------------------------------------------------------------------
# 8. Sync mode — full result
# ---------------------------------------------------------------------------


class TestSyncMode:
    """Default (sync) mode should return the full detection result."""

    def test_sync_returns_full_result(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection",
                return_value=_fake_result("real", 0.85, "image"),
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("img.png", _MINIMAL_PNG, "image/png")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["verdict"] == "real"
        assert body["confidence"] == 0.85
        assert body["media_type"] == "image"

    def test_sync_no_status_field(self, test_client):
        """Sync response must NOT include a 'status' or 'poll_url' field."""
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection", return_value=_fake_result()
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                files={"file": ("img.png", _MINIMAL_PNG, "image/png")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        assert "poll_url" not in body

    def test_sensitivity_balanced_accepted(self, test_client):
        """Sending sensitivity=balanced should not error."""
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection", return_value=_fake_result()
            ),
        ):
            response = test_client.post(
                "/v1/detect",
                data={"sensitivity": "balanced"},
                files={"file": ("img.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 9. Async mode — 202 with poll_url
# ---------------------------------------------------------------------------


class TestAsyncMode:
    """Async mode should return 202 with status=processing and a poll_url."""

    def test_async_returns_202(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._check_async_allowed", return_value=None),
            patch("services.detection._enqueue_async_detection", return_value=None),
        ):
            response = test_client.post(
                "/v1/detect",
                data={"async": "true"},
                files={"file": ("img.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 202

    def test_async_status_is_processing(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._check_async_allowed", return_value=None),
            patch("services.detection._enqueue_async_detection", return_value=None),
        ):
            response = test_client.post(
                "/v1/detect",
                data={"async": "true"},
                files={"file": ("img.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        assert body["status"] == "processing"

    def test_async_has_poll_url(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._check_async_allowed", return_value=None),
            patch("services.detection._enqueue_async_detection", return_value=None),
        ):
            response = test_client.post(
                "/v1/detect",
                data={"async": "true"},
                files={"file": ("img.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        assert "poll_url" in body
        assert body["poll_url"].startswith("/v1/results/")

    def test_async_id_has_det_prefix(self, test_client):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._check_async_allowed", return_value=None),
            patch("services.detection._enqueue_async_detection", return_value=None),
        ):
            response = test_client.post(
                "/v1/detect",
                data={"async": "true"},
                files={"file": ("img.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        body = response.json()
        assert body["id"].startswith("det_")

    def test_free_tier_async_returns_402(self, test_client):
        """Free-tier API keys must not use async mode."""
        api_key = _make_api_key(tier="free")

        from fastapi import HTTPException

        def _raise(*args, **kwargs):
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "async_not_allowed",
                    "message": "Async processing is not available on the free tier.",
                },
            )

        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._check_async_allowed", side_effect=_raise),
        ):
            response = test_client.post(
                "/v1/detect",
                data={"async": "true"},
                files={"file": ("img.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )
        assert response.status_code == 402
        body = response.json()
        detail = body.get("detail") or body
        assert detail.get("error") == "async_not_allowed"


# ---------------------------------------------------------------------------
# 10. Response schema validation
# ---------------------------------------------------------------------------


class TestResponseSchema:
    """Sync response must include all required fields with correct types."""

    def _call_detect(self, test_client, verdict="fake", confidence=0.93):
        api_key = _make_api_key(tier="starter")
        with (
            patch("routers.v1._authenticate", return_value=_make_auth_result(api_key)),
            patch("routers.v1._check_rate_limit", return_value=None),
            patch("routers.v1._check_monthly_quota", return_value=None),
            patch("routers.v1._increment_usage", return_value=None),
            patch(
                "services.detection._run_sync_detection",
                return_value=_fake_result(verdict, confidence, "image"),
            ),
        ):
            return test_client.post(
                "/v1/detect",
                files={"file": ("photo.jpg", _MINIMAL_JPEG, "image/jpeg")},
                headers={"Authorization": "Bearer ds_live_valid"},
            )

    def test_id_is_string(self, test_client):
        body = self._call_detect(test_client).json()
        assert isinstance(body["id"], str)

    def test_verdict_is_string(self, test_client):
        body = self._call_detect(test_client).json()
        assert isinstance(body["verdict"], str)

    def test_confidence_is_float(self, test_client):
        body = self._call_detect(test_client, confidence=0.93).json()
        assert isinstance(body["confidence"], float)

    def test_media_type_is_string(self, test_client):
        body = self._call_detect(test_client).json()
        assert isinstance(body["media_type"], str)

    def test_created_at_is_string(self, test_client):
        body = self._call_detect(test_client).json()
        assert isinstance(body["created_at"], str)

    def test_all_required_fields_present(self, test_client):
        body = self._call_detect(test_client).json()
        for field in ("id", "verdict", "confidence", "media_type", "created_at"):
            assert field in body, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# 11. File too large — 413
# ---------------------------------------------------------------------------


class TestFileTooLarge:
    """Requests whose Content-Length exceeds MAX_GENERAL_PAYLOAD_SIZE_BYTES return 413.

    The middleware checks the Content-Length header before the route handler
    runs, so we can set a large header without sending the actual bytes.
    """

    def test_oversized_content_length_returns_413(self, test_client):
        # MAX_GENERAL_PAYLOAD_SIZE_BYTES = (100 + 15) * 1024 * 1024 = 120 MB
        # Send a header claiming 200 MB.
        oversized = str(200 * 1024 * 1024)
        response = test_client.post(
            "/v1/detect",
            files={"file": ("huge.jpg", _MINIMAL_JPEG, "image/jpeg")},
            headers={
                "Authorization": "Bearer ds_live_valid",
                "Content-Length": oversized,
            },
        )
        assert response.status_code == 413

    def test_413_response_contains_size_info(self, test_client):
        oversized = str(200 * 1024 * 1024)
        response = test_client.post(
            "/v1/detect",
            files={"file": ("huge.jpg", _MINIMAL_JPEG, "image/jpeg")},
            headers={
                "Authorization": "Bearer ds_live_valid",
                "Content-Length": oversized,
            },
        )
        body = response.json()
        # The middleware returns {"detail": "...", "request_id": "..."}.
        assert "detail" in body


class TestAuthDisabled:
    """With auth disabled, the API serves a local user instead of rejecting."""

    @pytest.fixture(autouse=True)
    def _disable_auth(self, monkeypatch):
        import routers.v1 as v1

        monkeypatch.setattr(v1, "REQUIRE_AUTH", False)

    def test_no_header_is_not_rejected(self, test_client):
        """A self-hosted install must not need credentials to call the API."""
        response = test_client.post(
            "/v1/detect",
            files={"file": ("photo.jpg", b"data", "image/jpeg")},
        )
        assert response.status_code != 401

    def test_usage_reports_an_unmetered_plan(self, test_client):
        """Quota fields are null, not zero, when nothing is being metered."""
        response = test_client.get("/v1/usage")
        assert response.status_code == 200
        body = response.json()
        assert body["scans_limit"] is None
        assert body["scans_remaining"] is None
