"""Put src/ and scripts/ on sys.path so tests run from a plain checkout (no pip install needed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT/"scripts", ROOT/"src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
