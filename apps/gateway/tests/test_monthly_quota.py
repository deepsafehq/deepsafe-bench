"""Tests for monthly quota reset logic in v1_router."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from routers.v1 import _check_monthly_quota, _maybe_reset_monthly_quota

_SENTINEL = object()


def _make_api_key(tier="free", scans_used=0, period_start=_SENTINEL):
    """Build a mock ApiKey with sensible defaults.

    Args:
        tier: Subscription tier name.
        scans_used: Current scan count on the key.
        period_start: Billing period start.  Pass ``None`` explicitly to
            set the attribute to ``None``; omit to use ``now(UTC)``.

    Returns:
        A MagicMock configured to look like an ApiKey row.
    """
    key = MagicMock()
    key.id = "test-key-id"
    key.user_id = "test-user-id"
    key.tier = tier
    key.scans_used = scans_used
    key.period_start = (
        datetime.now(timezone.utc) if period_start is _SENTINEL else period_start
    )
    key.revoked_at = None
    return key


class TestMaybeResetMonthlyQuota:
    """Unit tests for _maybe_reset_monthly_quota."""

    def test_free_tier_never_resets(self):
        """Free tier has is_monthly=False, so quota must never be reset."""
        key = _make_api_key(tier="free", scans_used=150)
        db = MagicMock()
        _maybe_reset_monthly_quota(key, db)
        db.query.assert_not_called()

    def test_monthly_tier_resets_after_30_days(self):
        """Starter tier resets scans_used when 30+ days have elapsed."""
        old_start = datetime.now(timezone.utc) - timedelta(days=31)
        key = _make_api_key(tier="starter", scans_used=1500, period_start=old_start)
        db = MagicMock()
        mock_filtered = MagicMock()
        db.query.return_value.filter.return_value = mock_filtered
        _maybe_reset_monthly_quota(key, db)
        mock_filtered.update.assert_called_once()
        assert key.scans_used == 0

    def test_monthly_tier_no_reset_within_period(self):
        """Starter tier must NOT reset when fewer than 30 days elapsed."""
        recent_start = datetime.now(timezone.utc) - timedelta(days=10)
        key = _make_api_key(tier="starter", scans_used=500, period_start=recent_start)
        db = MagicMock()
        _maybe_reset_monthly_quota(key, db)
        db.query.assert_not_called()
        assert key.scans_used == 500

    def test_none_period_start_does_not_crash(self):
        """A monthly key with period_start=None must not crash."""
        key = _make_api_key(tier="starter", period_start=None)
        db = MagicMock()
        _maybe_reset_monthly_quota(key, db)
        db.query.assert_not_called()


class TestCheckMonthlyQuota:
    """Unit tests for _check_monthly_quota."""

    def test_free_tier_quota_exceeded_raises_402(self):
        """Hitting the 200-scan free limit must raise HTTP 402."""
        key = _make_api_key(tier="free", scans_used=200)
        db = MagicMock()
        with patch("routers.v1._get_user_total_scans", return_value=200):
            with pytest.raises(HTTPException) as exc_info:
                _check_monthly_quota(key, db)
        assert exc_info.value.status_code == 402

    def test_within_quota_does_not_raise(self):
        """Usage below the free-tier limit must not raise."""
        key = _make_api_key(tier="free", scans_used=10)
        db = MagicMock()
        with patch("routers.v1._get_user_total_scans", return_value=10):
            _check_monthly_quota(key, db)

    def test_unlimited_tier_never_raises(self):
        """An unrecognised tier (no monthly limit) must not raise."""
        key = _make_api_key(tier="nonexistent")
        db = MagicMock()
        with patch("routers.v1._get_user_total_scans", return_value=99999):
            _check_monthly_quota(key, db)
