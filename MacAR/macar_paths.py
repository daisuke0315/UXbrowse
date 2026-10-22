"""Code and local research data have separate roots; no legacy path fallback."""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('VIEWSYNC_MACAR_DATA_DIR',
                               CODE_DIR.parent/'data'/'MacAR')).expanduser().resolve()
