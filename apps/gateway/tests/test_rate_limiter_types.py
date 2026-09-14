"""Tests for RateLimitInfo TypedDict and get_remaining return annotation."""

import typing

from rate_limiter import RateLimiter


def test_get_remaining_has_typed_return():
    from rate_limiter import RateLimitInfo

    hints = typing.get_type_hints(RateLimiter.get_remaining)
    assert "return" in hints, "get_remaining must have a return type annotation"
    assert (
        hints["return"] is RateLimitInfo
    ), "get_remaining return type must be RateLimitInfo"


def test_rate_limit_info_keys():
    from rate_limiter import RateLimitInfo

    assert "limit" in RateLimitInfo.__annotations__
    assert "remaining" in RateLimitInfo.__annotations__
    assert "reset" in RateLimitInfo.__annotations__
