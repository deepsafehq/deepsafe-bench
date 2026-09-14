"""Tests for /api/keys/auto and /api/dashboard endpoints."""

import uuid

import pytest
from fastapi.testclient import TestClient


def get_client():
    """Return a fresh TestClient bound to the app."""
    from main import app

    return TestClient(app)


class TestAutoKey:
    """Tests for POST /api/keys/auto."""

    @pytest.fixture(autouse=True)
    def mock_auth(self):
        """Override JWT auth with a unique user ID per test to prevent DB state leakage."""
        from main import app, get_current_user

        unique_id = f"test-user-auto-{uuid.uuid4().hex}"

        async def mock_user():
            return {"sub": unique_id, "email": "test@example.com"}

        app.dependency_overrides[get_current_user] = mock_user
        yield
        app.dependency_overrides = {}

    def test_auto_key_creates_on_first_call(self):
        """First call should create a key and return the plaintext."""
        client = get_client()
        response = client.post("/api/keys/auto")
        assert response.status_code == 200
        data = response.json()
        assert data["created"] is True
        assert "key" in data
        assert data["key"].startswith("ds_live_")
        assert data["tier"] == "free"
        assert data["scans_used"] == 0
        assert data["scans_limit"] == 200

    def test_auto_key_returns_existing_on_second_call(self):
        """Second call should return metadata only — no plaintext key."""
        client = get_client()
        client.post("/api/keys/auto")
        response = client.post("/api/keys/auto")
        assert response.status_code == 200
        data = response.json()
        assert data["created"] is False
        assert "key" not in data
        assert "key_prefix" in data

    def test_auto_key_response_has_expected_fields_on_create(self):
        """Created response must include id, tier, scans_used, scans_limit, key_prefix."""
        client = get_client()
        response = client.post("/api/keys/auto")
        data = response.json()
        for field in ("id", "tier", "scans_used", "scans_limit", "key_prefix"):
            assert field in data, f"Missing field: {field}"

    def test_auto_key_requires_auth(self):
        """Without auth override the endpoint must reject unauthenticated requests."""
        from main import app

        # Ensure no override is active
        app.dependency_overrides = {}
        client = get_client()
        response = client.post("/api/keys/auto")
        assert response.status_code in {401, 403}


class TestDashboardEndpoint:
    """Tests for GET /api/dashboard."""

    @pytest.fixture(autouse=True)
    def mock_auth(self):
        """Override JWT auth with a unique user ID per test to prevent DB state leakage."""
        from main import app, get_current_user

        unique_id = f"test-user-dashboard-{uuid.uuid4().hex}"

        async def mock_user():
            return {"sub": unique_id, "email": "test@example.com"}

        app.dependency_overrides[get_current_user] = mock_user
        yield
        app.dependency_overrides = {}

    def test_dashboard_returns_plan_and_detections(self):
        """Dashboard must return plan info and recent_detections list."""
        client = get_client()
        client.post("/api/keys/auto")
        response = client.get("/api/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "plan" in data
        assert data["plan"]["tier"] == "free"
        assert data["plan"]["scans_limit"] == 200
        assert data["plan"]["is_monthly"] is False
        assert "detections" in data
        assert isinstance(data["detections"], list)

    def test_dashboard_plan_has_required_fields(self):
        """Plan object must include tier, scans_used, scans_limit, is_monthly."""
        client = get_client()
        client.post("/api/keys/auto")
        response = client.get("/api/dashboard")
        plan = response.json()["plan"]
        for field in ("tier", "scans_used", "scans_limit", "is_monthly"):
            assert field in plan, f"Missing plan field: {field}"

    def test_dashboard_no_key_still_returns_defaults(self):
        """If user has no key, dashboard should return free-tier defaults."""
        client = get_client()
        # Do NOT call /api/keys/auto first
        response = client.get("/api/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "plan" in data
        assert "detections" in data

    def test_dashboard_requires_auth(self):
        """Without auth the endpoint must reject unauthenticated requests."""
        from main import app

        app.dependency_overrides = {}
        client = get_client()
        response = client.get("/api/dashboard")
        assert response.status_code in {401, 403}
