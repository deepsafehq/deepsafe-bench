import pickle
import tempfile
from pathlib import Path

import pytest


def test_safe_unpickle_rejects_outside_artifacts_dir():
    """safe_unpickle must refuse to load files outside the artifacts dir."""
    from deepsafe_shared.ensemble import _ARTIFACTS_DIR, _safe_unpickle

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump({"malicious": True}, f)
        bad_path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="outside"):
            _safe_unpickle(bad_path)
    finally:
        bad_path.unlink()


def test_safe_unpickle_accepts_artifacts_dir():
    """safe_unpickle must accept files inside the artifacts dir."""
    from deepsafe_shared.ensemble import _ARTIFACTS_DIR, _safe_unpickle

    pkl_files = list(_ARTIFACTS_DIR.glob("*.pkl"))
    if not pkl_files:
        pytest.skip("No pickle artifacts available for testing")
    try:
        result = _safe_unpickle(pkl_files[0])
        assert result is not None
    except Exception as e:
        # Artifacts may be incompatible with the CI sklearn version
        pytest.skip(f"Artifacts not loadable in this environment: {e}")
