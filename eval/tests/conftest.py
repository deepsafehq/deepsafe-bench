"""Ensure eval and scripts packages are importable when running tests."""

import sys
from pathlib import Path

# Add eval/ to path so `from deepsafe_eval import ...` works
eval_dir = Path(__file__).parent.parent
if str(eval_dir) not in sys.path:
    sys.path.insert(0, str(eval_dir))

# Add project root so `from eval.*` and `from scripts.*` imports work
project_root = eval_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
