"""Redis-based rate limiting for the public API.

Uses atomic Lua scripts so that INCR + EXPIRE cannot diverge if the
process crashes between the two calls.  The TTL is set only on the
first request in each window (fixed window, not sliding).
"""

import time
from datetime import date
from typing import TypedDict


class RateLimitInfo(TypedDict):
    """Rate limit status for response headers."""

    limit: int
    remaining: int
    reset: int


class RateLimitExceeded(Exception):
    """Raised when a rate limit is exceeded."""

    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s.")


TIER_LIMITS = {
    "free": {"rpm": 5, "daily": 10, "monthly": 200, "is_monthly": False},
    "starter": {"rpm": 20, "daily": None, "monthly": 2000, "is_monthly": True},
    "pro": {"rpm": 60, "daily": None, "monthly": 10000, "is_monthly": True},
}

# Lua script: atomic INCR + EXPIRE-on-first-request (fixed window).
# KEYS[1] = rate-limit key, ARGV[1] = TTL in seconds.
# Returns the new counter value.
_LUA_RATE_LIMIT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""


class RateLimiter:
    """Check per-minute and daily rate limits via Redis."""

    def __init__(self, redis_client, tier_limits=None):
        self._redis = redis_client
        self._limits = tier_limits or TIER_LIMITS
        self._lua_script = self._redis.register_script(_LUA_RATE_LIMIT)

    def check(self, identifier: str, tier: str) -> None:
        """Raise RateLimitExceeded if any limit is breached.

        Args:
            identifier: User ID or key hash to namespace Redis keys.
            tier: Subscription tier name (e.g. 'free', 'starter', 'pro').

        Raises:
            RateLimitExceeded: If the rpm or daily limit for the tier
                is exceeded.
        """
        limits = self._limits[tier]
        rpm_limit = limits["rpm"]
        daily_limit = limits.get("daily")

        rpm_key = f"ratelimit:{identifier}:rpm"
        rpm_count = self._lua_script(keys=[rpm_key], args=[60])

        if rpm_count > rpm_limit:
            raise RateLimitExceeded(retry_after=60)

        if daily_limit is not None:
            today = date.today().isoformat()
            daily_key = f"ratelimit:{identifier}:daily:{today}"
            # 48 h cleanup buffer; TTL set only on the first request.
            daily_count = self._lua_script(
                keys=[daily_key],
                args=[172800],
            )
            if daily_count > daily_limit:
                raise RateLimitExceeded(retry_after=3600)

    def get_remaining(self, key_hash: str, tier: str) -> RateLimitInfo:
        """Return current limit state for response headers.

        Args:
            key_hash: Hashed API key used to namespace Redis keys.
            tier: Subscription tier name (e.g. 'free', 'starter', 'pro').

        Returns:
            A dict with 'limit', 'remaining', and 'reset' (Unix
            timestamp) keys.
        """
        limits = self._limits[tier]
        rpm_key = f"ratelimit:{key_hash}:rpm"
        current = self._redis.get(rpm_key)
        count = int(current) if current else 0
        ttl = self._redis.ttl(rpm_key)
        reset_at = int(time.time()) + max(ttl, 0)
        return RateLimitInfo(
            limit=limits["rpm"],
            remaining=max(limits["rpm"] - count, 0),
            reset=reset_at,
        )
