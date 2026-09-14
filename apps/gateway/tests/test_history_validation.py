"""Tests for media_type validation in the history endpoint."""

from routers.history import _VALID_MEDIA_TYPES


def test_valid_media_types_constant():
    """Ensure the set of accepted media types is correct."""
    assert _VALID_MEDIA_TYPES == {"image", "video", "audio"}
