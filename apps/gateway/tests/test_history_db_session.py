import inspect


def test_history_no_raw_session_local():
    """history.py should not use raw SessionLocal() calls."""
    from routers import history

    source = inspect.getsource(history)
    lines = source.split("\n")
    raw_session_lines = [
        line for line in lines if "SessionLocal()" in line and "import" not in line
    ]
    assert len(raw_session_lines) == 0, (
        f"Found {len(raw_session_lines)} raw SessionLocal() calls. "
        "Use Depends(get_db) or get_db_session() instead."
    )
