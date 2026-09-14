"""Tests for query_model_api error handling paths."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from services.detection import query_model_api


@pytest.fixture(autouse=True)
def mock_config():
    """Provide a minimal model config for all tests."""
    config = {
        "media_types": {
            "image": {
                "model_endpoints": {
                    "test_model": "http://localhost:9001/predict",
                },
            },
        },
    }
    with patch("services.detection.ALL_MODEL_CONFIGS", config):
        with patch("services.detection.MAX_RETRIES", 1):
            with patch("services.detection.DEFAULT_TIMEOUT", 5):
                yield


def test_unconfigured_model_returns_error():
    result = query_model_api("nonexistent", "image", "base64data", 0.5, "req-1")
    assert "error" in result


def test_timeout_returns_error_after_retries():
    with patch("services.detection.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        result = query_model_api("test_model", "image", "base64data", 0.5, "req-1")
    assert "error" in result
    assert "timed out" in result["error"].lower()


def test_http_500_returns_error():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    with patch("services.detection.requests.post", return_value=mock_response):
        result = query_model_api("test_model", "image", "base64data", 0.5, "req-1")
    assert "error" in result


def test_connection_error_returns_error():
    with patch("services.detection.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        result = query_model_api("test_model", "image", "base64data", 0.5, "req-1")
    assert "error" in result
    assert "unreachable" in result["error"].lower()


def test_invalid_json_returns_error():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = json.JSONDecodeError("bad", "", 0)
    mock_response.text = "not json"
    with patch("services.detection.requests.post", return_value=mock_response):
        result = query_model_api("test_model", "image", "base64data", 0.5, "req-1")
    assert "error" in result
    assert "invalid" in result["error"].lower()


def test_successful_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"probability": 0.85, "prediction": 1}
    with patch("services.detection.requests.post", return_value=mock_response):
        result = query_model_api("test_model", "image", "base64data", 0.5, "req-1")
    assert result["probability"] == 0.85
    assert "error" not in result


def test_missing_payload_key_returns_error():
    result = query_model_api("test_model", "unknown_type", "base64data", 0.5, "req-1")
    assert "error" in result
