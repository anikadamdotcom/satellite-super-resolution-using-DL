"""Project folders (demo_images/, checkpoints/) located from the source checkout."""
from pathlib import Path

# src/satsr/paths.py -> project root. Assumes a source checkout or an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
