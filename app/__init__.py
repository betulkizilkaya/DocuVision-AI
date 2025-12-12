# app/__init__.py
from pathlib import Path

# Proje kök dizini
ROOT_DIR = Path(__file__).resolve().parents[1]

# Sık kullanılan yollar
DATA_DIR = ROOT_DIR / "data"
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"

from app.core import db  # istersen komple modülü de export edebilirsin

# Text tarafı
from app.text.doc_classifier import (
    predict_doc_type,
    load_model,
)

from app.text.text_ops import extract_text_lines
