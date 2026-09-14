import pytest


def test_err_helper_produces_correct_shape():
    """The _err helper should produce {error, message} dicts."""
    from routers.v1 import _err

    exc = _err("not_found", "Resource not found.", 404)
    assert exc.status_code == 404
    assert exc.detail == {"error": "not_found", "message": "Resource not found."}


def test_history_404_uses_dict_detail():
    """All HTTPException details in history.py should use dict format."""
    import inspect

    from routers import history

    source = inspect.getsource(history)
    # Should not have bare f-string detail for 404s
    # All detail= should use dict format
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        if 'detail=f"' in line or "detail=f'" in line:
            pytest.fail(f"Line {i} uses bare f-string detail: {line.strip()}")
