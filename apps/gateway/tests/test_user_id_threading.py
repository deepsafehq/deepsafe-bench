"""Tests for user_id threading through detection pipeline."""

import inspect

from main import run_detection


def test_run_detection_accepts_user_id():
    """run_detection task must accept a user_id parameter."""
    sig = inspect.signature(run_detection)
    param_names = list(sig.parameters.keys())
    assert (
        "user_id" in param_names
    ), f"run_detection missing user_id param. Has: {param_names}"
