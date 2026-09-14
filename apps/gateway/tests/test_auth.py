from datetime import datetime, timedelta

import jwt as pyjwt
import pytest
from fastapi import HTTPException

MOCK_JWT_SECRET = "test-secret-key-for-testing-only"


def make_test_jwt(user_id: str = "test-uuid-123", expired: bool = False):
    """Create a test Supabase-style JWT."""
    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "email": "test@example.com",
        "iss": "https://test.supabase.co/auth/v1",
        "aud": "authenticated",
        "iat": now,
        "exp": now + timedelta(hours=-1 if expired else 1),
    }
    return pyjwt.encode(payload, MOCK_JWT_SECRET, algorithm="HS256")


def test_verify_valid_jwt():
    from auth import verify_supabase_jwt

    token = make_test_jwt(user_id="abc-123")
    result = verify_supabase_jwt(token, jwt_secret=MOCK_JWT_SECRET)
    assert result["sub"] == "abc-123"
    assert result["email"] == "test@example.com"


def test_verify_expired_jwt():
    from auth import verify_supabase_jwt

    token = make_test_jwt(expired=True)
    with pytest.raises(HTTPException):
        verify_supabase_jwt(token, jwt_secret=MOCK_JWT_SECRET)


def test_verify_invalid_jwt():
    from auth import verify_supabase_jwt

    with pytest.raises(HTTPException):
        verify_supabase_jwt("not.a.jwt", jwt_secret=MOCK_JWT_SECRET)


def test_verify_missing_jwt_secret():
    from auth import verify_supabase_jwt

    with pytest.raises(HTTPException) as exc_info:
        verify_supabase_jwt("some.token.here", jwt_secret="")
    # With no valid secret, server is misconfigured — returns 500.
    assert exc_info.value.status_code == 500
