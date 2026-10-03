"""Allow running the tool with ``python -m decigrep``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())