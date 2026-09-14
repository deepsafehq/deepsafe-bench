"""Tests for the Redis-based rate limiter."""

from datetime import date

import pytest
from rate_limiter import TIER_LIMITS, RateLimiter, RateLimitExceeded


class FakeScript:
    """Simulates the atomic Lua INCR-and-EXPIRE-on-first-request script.

    Mirrors ``_LUA_RATE_LIMIT``: increments the key and sets a TTL only
    when the counter transitions from 0 -> 1 (first request in the
    fixed window).
    """

    def __init__(self, redis):
        self._redis = redis

    def __call__(self, keys, args):
        key = keys[0]
        ttl = int(args[0])
        val = self._redis._store.get(key, 0) + 1
        self._redis._store[key] = val
        if val == 1:
            self._redis._ttls[key] = ttl
        return val


class FakeRedis:
    """Minimal Redis mock that supports ``register_script``."""

    def __init__(self):
        self._store = {}
        self._ttls = {}

    def register_script(self, _lua_src: str):
        """Return a callable that mimics the Lua script."""
        return FakeScript(self)

    def get(self, key):
        return self._store.get(key)

    def ttl(self, key):
        return self._ttls.get(key, -2)


class TestRateLimiterRPM:
    """Per-minute rate limit enforcement."""

    def test_first_request_passes(self):
        limiter = RateLimiter(FakeRedis())
        limiter.check("user-1", "free")

    def test_within_limit_passes(self):
        limiter = RateLimiter(FakeRedis())
        for _ in range(5):
            limiter.check("user-1", "free")

    def test_exceeding_rpm_raises(self):
        limiter = RateLimiter(FakeRedis())
        for _ in range(5):
            limiter.check("user-1", "free")
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check("user-1", "free")
        assert exc_info.value.retry_after == 60

    def test_different_users_independent(self):
        limiter = RateLimiter(FakeRedis())
        for _ in range(5):
            limiter.check("user-1", "free")
        limiter.check("user-2", "free")

    def test_pro_tier_higher_limit(self):
        limiter = RateLimiter(FakeRedis())
        for _ in range(60):
            limiter.check("user-1", "pro")
        with pytest.raises(RateLimitExceeded):
            limiter.check("user-1", "pro")


class TestRateLimiterDaily:
    """Daily rate limit enforcement (free tier only)."""

    def test_daily_limit_enforced(self):
        redis = FakeRedis()
        limiter = RateLimiter(redis)
        for i in range(10):
            redis._store["ratelimit:user-1:rpm"] = 0
            limiter.check("user-1", "free")
        redis._store["ratelimit:user-1:rpm"] = 0
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check("user-1", "free")
        assert exc_info.value.retry_after == 3600

    def test_starter_tier_no_daily_limit(self):
        limiter = RateLimiter(FakeRedis())
        for _ in range(20):
            limiter.check("user-1", "starter")


class TestDailyKeyFormat:
    """Daily keys use date-based naming (no sliding window)."""

    def test_daily_key_contains_date(self):
        redis = FakeRedis()
        RateLimiter(redis).check("user-1", "free")
        today = date.today().isoformat()
        assert f"ratelimit:user-1:daily:{today}" in redis._store


class TestTierConfig:
    """Tier configuration constants."""

    def test_free_tier(self):
        assert TIER_LIMITS["free"]["rpm"] == 5
        assert TIER_LIMITS["free"]["monthly"] == 200
        assert TIER_LIMITS["free"]["is_monthly"] is False

    def test_starter_tier(self):
        assert TIER_LIMITS["starter"]["monthly"] == 2000
        assert TIER_LIMITS["starter"]["is_monthly"] is True

    def test_pro_tier(self):
        assert TIER_LIMITS["pro"]["rpm"] == 60
        assert TIER_LIMITS["pro"]["monthly"] == 10000
        assert TIER_LIMITS["pro"]["is_monthly"] is True


class TestGetRemaining:
    """get_remaining() returns correct header values."""

    def test_full_remaining_no_requests(self):
        limiter = RateLimiter(FakeRedis())
        result = limiter.get_remaining("user-1", "free")
        assert result["limit"] == 5
        assert result["remaining"] == 5

    def test_remaining_decreases(self):
        redis = FakeRedis()
        limiter = RateLimiter(redis)
        limiter.check("user-1", "free")
        assert limiter.get_remaining("user-1", "free")["remaining"] == 4

    def test_remaining_never_negative(self):
        redis = FakeRedis()
        redis._store["ratelimit:user-1:rpm"] = 100
        assert RateLimiter(redis).get_remaining("user-1", "free")["remaining"] == 0
