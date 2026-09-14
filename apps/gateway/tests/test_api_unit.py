import pytest
from fastapi.testclient import TestClient
from main import app, get_current_user

client = TestClient(app)


# Mock auth dependency to return valid Supabase JWT payload
async def mock_get_current_user():
    return {
        "sub": "test-user-unit",
        "email": "test@example.com",
        "aud": "authenticated",
    }


@pytest.fixture
def mock_auth():
    app.dependency_overrides[get_current_user] = mock_get_current_user
    yield
    app.dependency_overrides = {}


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the DeepSafe" in response.json()["message"]


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("healthy", "degraded")
