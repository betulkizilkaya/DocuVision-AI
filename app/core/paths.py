# app/core/paths.py
from pathlib import Path

# Bu dosya: app/core/paths.py
# parents[2] = proje kökü (DocuVision-AI)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"               # dışarıdaki data/
DB_PATH  = ROOT_DIR / "db" / "corpus.sqlite"  # dışarıdaki db/corpus.sqlite

# Klasör garanti
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
