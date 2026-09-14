"""Tests for v1_router authentication helper functions."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from rate_limiter import RateLimitExceeded
from routers.v1 import (
    AuthResult,
    _check_async_allowed,
    _check_rate_limit,
    _get_auth_header,
    _validate_file_content,
)


class TestGetAuthHeader:
    def test_valid_bearer_token(self):
        request = MagicMock()
        request.headers = {"Authorization": "Bearer test-token-123"}
        assert _get_auth_header(request) == "test-token-123"

    def test_missing_header_raises_401(self):
        request = MagicMock()
        request.headers = {}
        with pytest.raises(HTTPException) as exc_info:
            _get_auth_header(request)
        assert exc_info.value.status_code == 401

    def test_non_bearer_header_raises_401(self):
        request = MagicMock()
        request.headers = {"Authorization": "Basic abc123"}
        with pytest.raises(HTTPException) as exc_info:
            _get_auth_header(request)
        assert exc_info.value.status_code == 401

    def test_empty_bearer_raises_401(self):
        request = MagicMock()
        request.headers = {"Authorization": "Bearer "}
        with pytest.raises(HTTPException) as exc_info:
            _get_auth_header(request)
        assert exc_info.value.status_code == 401


class TestCheckAsyncAllowed:
    def test_sync_always_allowed(self):
        auth = AuthResult(api_key=MagicMock(tier="free"), is_jwt_auth=False)
        _check_async_allowed(auth, is_async=False)

    def test_jwt_auth_always_gets_async(self):
        auth = AuthResult(api_key=MagicMock(tier="free"), is_jwt_auth=True)
        _check_async_allowed(auth, is_async=True)

    def test_free_tier_api_key_no_async(self):
        auth = AuthResult(api_key=MagicMock(tier="free"), is_jwt_auth=False)
        with pytest.raises(HTTPException) as exc_info:
            _check_async_allowed(auth, is_async=True)
        assert exc_info.value.status_code == 402

    def test_paid_tier_api_key_gets_async(self):
        auth = AuthResult(api_key=MagicMock(tier="starter"), is_jwt_auth=False)
        _check_async_allowed(auth, is_async=True)


class TestCheckRateLimit:
    def test_passes_when_under_limit(self):
        api_key = MagicMock(user_id="user-1", tier="free")
        limiter = MagicMock()
        limiter.check.return_value = None
        _check_rate_limit(api_key, MagicMock(), limiter)

    def test_raises_429_when_rate_limited(self):
        api_key = MagicMock(user_id="user-1", tier="free")
        limiter = MagicMock()
        limiter.check.side_effect = RateLimitExceeded(retry_after=60)
        with pytest.raises(HTTPException) as exc_info:
            _check_rate_limit(api_key, MagicMock(), limiter)
        assert exc_info.value.status_code == 429


class TestValidateFileContent:
    def test_jpeg_matches_image(self):
        file_bytes = b"\xff\xd8\xff" + b"\x00" * 100
        _validate_file_content(file_bytes, "image")

    def test_jpeg_mismatches_audio(self):
        file_bytes = b"\xff\xd8\xff" + b"\x00" * 100
        with pytest.raises(HTTPException) as exc_info:
            _validate_file_content(file_bytes, "audio")
        assert exc_info.value.status_code == 415

    def test_too_small_file_passes(self):
        _validate_file_content(b"\x00\x01\x02", "image")

    def test_unknown_magic_passes(self):
        file_bytes = b"\xab\xcd\xef\x00" + b"\x00" * 100
        _validate_file_content(file_bytes, "image")
