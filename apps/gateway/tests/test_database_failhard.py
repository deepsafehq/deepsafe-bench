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
    """Production without DATABASE_URL must refuse to start.

    Outside production the gateway falls back to a local SQLite file so that a
    fresh clone runs with no configuration; this test pins the guarantee that
    a real deployment cannot silently do the same.
    """
    import database

    # Build an environment that has no DATABASE_URL and no pytest markers.
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DATABASE_URL", "PYTEST_CURRENT_TEST")
    }
    # Also strip "pytest" from the "_" env var (the parent process path).
    clean_env["_"] = "/usr/bin/python"
    clean_env["DEEPSAFE_ENV"] = "production"

    # Temporarily hide "pytest" from sys.modules so the sys.modules check
    # doesn't short-circuit the detection.
    import config

    saved_pytest = sys.modules.pop("pytest", None)
    try:
        with mock.patch.dict(os.environ, clean_env, clear=True):
            # config resolves IS_PRODUCTION at import time, so a fresh-process
            # simulation has to reload it before database reads it.
            importlib.reload(config)
            with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
                importlib.reload(database)
    finally:
        if saved_pytest is not None:
            sys.modules["pytest"] = saved_pytest
        importlib.reload(config)
        importlib.reload(database)


def test_non_production_falls_back_to_sqlite():
    """Without DATABASE_URL and outside production, SQLite is used, not an error."""
    import database

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DATABASE_URL", "PYTEST_CURRENT_TEST")
    }
    clean_env["_"] = "/usr/bin/python"
    clean_env["DEEPSAFE_ENV"] = "development"

    import config

    saved_pytest = sys.modules.pop("pytest", None)
    try:
        with mock.patch.dict(os.environ, clean_env, clear=True):
            importlib.reload(config)
            importlib.reload(database)
            assert database.DATABASE_URL.startswith("sqlite")
    finally:
        if saved_pytest is not None:
            sys.modules["pytest"] = saved_pytest
        importlib.reload(config)
        importlib.reload(database)
