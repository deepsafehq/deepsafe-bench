"""Shared authentication helper utilities."""


def extract_user_id(current_user: dict) -> str:
    """Extract the user ID from a decoded JWT payload.

    Args:
        current_user: Decoded JWT payload dict from get_current_user.

    Returns:
        The Supabase user UUID from the 'sub' claim.

    Raises:
        ValueError: If the 'sub' claim is missing or empty.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise ValueError("JWT payload missing required 'sub' claim")
    return user_id
