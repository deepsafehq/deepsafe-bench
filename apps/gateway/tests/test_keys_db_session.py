import inspect


def test_keys_uses_get_db_not_raw_session():
    """keys.py should use Depends(get_db), not raw SessionLocal()."""
    from routers import keys

    source = inspect.getsource(keys)
    lines = source.split("\n")
    raw_session_lines = [
        line for line in lines if "SessionLocal()" in line and "import" not in line
    ]
    assert len(raw_session_lines) == 0, (
        f"Found {len(raw_session_lines)} raw SessionLocal() calls. "
        "Use Depends(get_db) instead."
    )
