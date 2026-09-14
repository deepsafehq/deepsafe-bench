"""Tests for config validation and env var overrides."""

import inspect

import config
import pytest
from config import DEFAULT_TIMEOUT, MAX_RETRIES


def test_config_default_timeout_is_positive():
    assert isinstance(DEFAULT_TIMEOUT, int)
    assert DEFAULT_TIMEOUT > 0


def test_config_max_retries_is_non_negative():
    assert isinstance(MAX_RETRIES, int)
    assert MAX_RETRIES >= 0


def test_config_timeout_env_override_referenced():
    source = inspect.getsource(config)
    assert "DEEPSAFE_DEFAULT_TIMEOUT" in source


def test_config_retries_env_override_referenced():
    source = inspect.getsource(config)
    assert "DEEPSAFE_MAX_RETRIES" in source


def test_supported_media_types_are_valid():
    from config import SUPPORTED_MEDIA_TYPES

    valid = {"image", "video", "audio", "provenance"}
    for mt in SUPPORTED_MEDIA_TYPES:
        assert mt in valid, f"Unknown media type: {mt}"
