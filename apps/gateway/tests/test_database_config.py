"""Tests for database configuration and engine setup."""

import os
from unittest import mock


def _reload_database_module():
    """Force-reload the database module to pick up new env vars."""
    import importlib

    import database

    importlib.reload(database)
    return database


class TestDatabaseConfig:
    """Verify DATABASE_URL parsing and engine configuration."""

    def test_sqlite_fallback_under_pytest(self):
        """When DATABASE_URL is empty and pytest is running, use SQLite."""
        with mock.patch.dict(os.environ, {"DATABASE_URL": ""}, clear=False):
            db = _reload_database_module()
            assert db.DATABASE_URL.startswith("sqlite:///")

    def test_postgresql_url_adds_sslmode(self):
        """Supabase requires sslmode=require for all connections."""
        test_url = "postgresql://user:pass@host:5432/db"
        with mock.patch.dict(os.environ, {"DATABASE_URL": test_url}, clear=False):
            db = _reload_database_module()
            assert "sslmode=require" in db.DATABASE_URL

    def test_postgresql_url_preserves_existing_sslmode(self):
        """Don't double-add sslmode if already present."""
        test_url = "postgresql://user:pass@host:5432/db?sslmode=verify-full"
        with mock.patch.dict(os.environ, {"DATABASE_URL": test_url}, clear=False):
            db = _reload_database_module()
            assert db.DATABASE_URL.count("sslmode") == 1

    def test_postgresql_pool_size_is_moderate(self):
        """Pool size should be <= 5 for Supabase pooler compatibility."""
        test_url = "postgresql://user:pass@host:5432/db"
        with mock.patch.dict(os.environ, {"DATABASE_URL": test_url}, clear=False):
            db = _reload_database_module()
            assert db._engine_kwargs["pool_size"] <= 5
