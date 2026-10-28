"""Make the `satsr` package importable when the project is not pip-installed.

Every script imports this first, so `python scripts/<name>.py` works from a plain checkout.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT/"src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
