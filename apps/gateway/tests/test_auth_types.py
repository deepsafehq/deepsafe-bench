"""Tests for JwtPayload TypedDict and typed return annotations."""

import typing


def test_verify_supabase_jwt_has_typed_return():
    from auth import verify_supabase_jwt

    hints = typing.get_type_hints(verify_supabase_jwt)
    assert "return" in hints
    assert hints["return"] is not dict, "Should be JwtPayload, not bare dict"


def test_jwt_payload_has_sub_field():
    from auth import JwtPayload

    assert "sub" in JwtPayload.__annotations__
