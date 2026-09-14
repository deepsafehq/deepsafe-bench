"""Tests for analytics API endpoints."""

from fastapi.testclient import TestClient

# HTTPBearer returns 403 when the Authorization header is absent entirely,
# and 401 when a token is present but invalid.  The important invariant is
# that every endpoint rejects unauthenticated callers (status >= 400).
_UNAUTH_STATUSES = {401, 403}


def test_analytics_summary_requires_auth():
    """GET /analytics/summary must require authentication."""
    from main import app

    client = TestClient(app)
    response = client.get("/analytics/summary")
    assert response.status_code in _UNAUTH_STATUSES


def test_analytics_usage_requires_auth():
    """GET /analytics/usage must require authentication."""
    from main import app

    client = TestClient(app)
    response = client.get("/analytics/usage")
    assert response.status_code in _UNAUTH_STATUSES


def test_analytics_models_requires_auth():
    """GET /analytics/models must require authentication."""
    from main import app

    client = TestClient(app)
    response = client.get("/analytics/models")
    assert response.status_code in _UNAUTH_STATUSES


def test_analytics_costs_requires_auth():
    """GET /analytics/costs must require authentication."""
    from main import app

    client = TestClient(app)
    response = client.get("/analytics/costs")
    assert response.status_code in _UNAUTH_STATUSES


def test_analytics_export_requires_auth():
    """GET /analytics/export must require authentication."""
    from main import app

    client = TestClient(app)
    response = client.get("/analytics/export")
    assert response.status_code in _UNAUTH_STATUSES
