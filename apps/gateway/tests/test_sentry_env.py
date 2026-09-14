"""Tests that the Sentry DSN is read from an environment variable, not hardcoded."""

import pathlib
import re

# A Sentry DSN looks like https://<key>@<org>.ingest.<region>.sentry.io/<id>.
# Matching the shape rather than a specific project keeps this test useful for
# any fork without embedding one deployment's identifiers in the source.
_DSN_PATTERN = re.compile(r"https://[0-9a-f]+@[\w.-]*ingest[\w.-]*\.sentry\.io/\d+")


def _get_main_source() -> str:
    """Return the source text of apps/gateway/main.py."""
    main_path = pathlib.Path(__file__).resolve().parents[1] / "main.py"
    return main_path.read_text()


def test_no_hardcoded_sentry_dsn():
    """main.py must not contain a hardcoded Sentry DSN string."""
    source = _get_main_source()
    match = _DSN_PATTERN.search(source)
    assert match is None, (
        "Hardcoded Sentry DSN found in main.py -- use the SENTRY_DSN env var instead"
    )


def test_sentry_dsn_read_from_env():
    """main.py must reference the SENTRY_DSN environment variable."""
    source = _get_main_source()
    assert "SENTRY_DSN" in source, "main.py does not reference SENTRY_DSN env var"
