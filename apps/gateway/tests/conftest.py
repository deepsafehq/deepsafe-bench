"""Shared test fixtures and setup for the gateway test suite."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def initialize_test_database():
    """Ensure all DB tables exist before any test runs.

    The lifespan context manager calls init_db() when the app starts, but
    TestClient instances used without ``with`` do not trigger lifespan. This
    fixture guarantees the schema is created for the SQLite test database
    regardless of how TestClient is used.
    """
    import api_keys  # noqa: F401 — registers ApiKey model with Base
    from database import init_db

    init_db()
