"""Tests for DATABASE_URL fail-hard behavior in production."""

import importlib
import os
import sys
from unittest import mock

import pytest


def test_test_env_falls_back_to_sqlite():
    """In the test environment, DATABASE_URL should resolve to sqlite or postgresql."""
    import database

    assert database.DATABASE_URL, "DATABASE_URL should not be empty in test env"
    assert database.DATABASE_URL.startswith(
        ("sqlite", "postgresql")
    ), f"Unexpected DATABASE_URL scheme: {database.DATABASE_URL}"


def test_production_raises_without_database_url():
    """When DATABASE_URL is unset and we are NOT in a test env, RuntimeError must be raised."""
    import database

    # Build an environment that has no DATABASE_URL and no pytest markers.
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DATABASE_URL", "PYTEST_CURRENT_TEST")
    }
    # Also strip "pytest" from the "_" env var (the parent process path).
    clean_env["_"] = "/usr/bin/python"

    # Temporarily hide "pytest" from sys.modules so the sys.modules check
    # doesn't short-circuit the detection.
    saved_pytest = sys.modules.pop("pytest", None)
    try:
        with mock.patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
                importlib.reload(database)
    finally:
        # Restore pytest in sys.modules and reload the module for other tests.
        if saved_pytest is not None:
            sys.modules["pytest"] = saved_pytest
        importlib.reload(database)
