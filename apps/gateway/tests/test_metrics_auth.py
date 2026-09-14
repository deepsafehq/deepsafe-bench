"""Tests for metrics endpoint token authentication security.

Verifies that the metrics bearer token comparison uses constant-time
comparison (hmac.compare_digest) to prevent timing side-channel attacks.
"""

import inspect

import metrics


def test_metrics_uses_constant_time_compare():
    """The metrics module must use hmac.compare_digest for token comparison."""
    source = inspect.getsource(metrics)
    assert "compare_digest" in source, (
        "metrics.py must use hmac.compare_digest for constant-time " "token comparison"
    )


def test_metrics_does_not_use_equality_for_token():
    """The metrics module must not use == or != for bearer token comparison."""
    source = inspect.getsource(metrics)
    assert 'auth != f"Bearer' not in source, (
        "metrics.py must not use != for bearer token comparison "
        "(vulnerable to timing attacks)"
    )
    assert "auth ==" not in source, (
        "metrics.py must not use == for bearer token comparison "
        "(vulnerable to timing attacks)"
    )
