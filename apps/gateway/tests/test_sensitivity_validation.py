"""Tests for sensitivity parameter validation in /v1/detect."""

import pytest
from routers.v1 import _SENSITIVITY_THRESHOLDS


def test_valid_sensitivity_values():
    """Known sensitivity presets must exist in the threshold map."""
    assert "balanced" in _SENSITIVITY_THRESHOLDS
    assert "strict" in _SENSITIVITY_THRESHOLDS
    assert "sensitive" in _SENSITIVITY_THRESHOLDS


def test_sensitivity_thresholds_are_floats():
    """Every threshold must be a number in [0.0, 1.0]."""
    for name, value in _SENSITIVITY_THRESHOLDS.items():
        assert isinstance(value, (int, float))
        assert 0.0 <= value <= 1.0


def test_invalid_sensitivity_not_in_thresholds():
    """Arbitrary strings must not be accepted as valid sensitivities."""
    assert "aggressive" not in _SENSITIVITY_THRESHOLDS
    assert "maximum" not in _SENSITIVITY_THRESHOLDS
    assert "" not in _SENSITIVITY_THRESHOLDS
