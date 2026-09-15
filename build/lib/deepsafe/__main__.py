"""Allow ``python -m deepsafe``."""

from deepsafe.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
