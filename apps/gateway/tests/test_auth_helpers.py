"""Tests for the extract_user_id authentication helper."""

import pytest


def test_extract_user_id_returns_sub():
    from auth_helpers import extract_user_id

    result = extract_user_id({"sub": "user-123", "email": "test@test.com"})
    assert result == "user-123"


def test_extract_user_id_raises_on_missing_sub():
    from auth_helpers import extract_user_id

    with pytest.raises(ValueError, match="sub"):
        extract_user_id({"email": "test@test.com"})


def test_extract_user_id_raises_on_empty_sub():
    from auth_helpers import extract_user_id

    with pytest.raises(ValueError, match="sub"):
        extract_user_id({"sub": "", "email": "test@test.com"})
