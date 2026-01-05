import json
import hashlib
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from app.core.db import create_connection, create_tables  # db.py

# ---- PATHS ----
APP_DIR = Path(__file__).resolve().parent        # .../app/model
ROOT_DIR = APP_DIR.parent.parent                 # repo root

IMAGES_DIR = ROOT_DIR / "temp" / "chessboard_exports" / "chessboard"

MODEL_DIR = APP_DIR / "models"                   # .../app/model/models
MODEL_PATH = MODEL_DIR / "chess_model_v3.keras"
CLASS_INDICES_PATH = MODEL_DIR / "class_indices.json"

# ---- SETTINGS ----
IMG_SIZE = (64, 64)
VALID_EXT = (".png", ".jpg", ".jpeg")
UPSCALE = 2.5

CELL_MARGIN_RATIO = 0.08   # 0.06–0.12 arası dene
USE_EQUALIZE = True        # ikon setlerde genelde iyi
BATCH_COMMIT = 50

CLASS_TO_FEN = {
    "White_Pawn": "P", "White_Rook": "R", "White_Knight": "N", "White_Bishop": "B",
    "White_Queen": "Q", "White_King": "K",
    "Black_Pawn": "p", "Black_Rook": "r", "Black_Knight": "n", "Black_Bishop": "b",
    "Black_Queen": "q", "Black_King": "k",
    "Empty_Square": None
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def get_or_create_file_id(conn: sqlite3.Connection, img_path: Path, sha256: str) -> int:
    row = conn.execute("SELECT id FROM file_index WHERE sha256 = ? LIMIT 1", (sha256,)).fetchone()
    if row:
        return int(row[0])

    conn.execute(
        "INSERT INTO file_index (filename, path, sha256, doc_type) VALUES (?, ?, ?, ?)",
        (img_path.name, str(img_path.resolve()), sha256, "chessboard_image"),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

def preprocess_cell(cell_bgr: np.ndarray) -> np.ndarray:
    """Return model input shape (1, H, W, 1) float32 [0..1]."""
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    if USE_EQUALIZE:
        gray = cv2.equalizeHist(gray)

    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    return np.expand_dims(gray, axis=(0, -1))

def board_image_to_fen_board(img_path: Path, model, idx_to_class: dict) -> str:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Görüntü okunamadı: {img_path}")

    img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    h, w, _ = img.shape
    cell_h = h // 8
    cell_w = w // 8

    m = int(CELL_MARGIN_RATIO * min(cell_h, cell_w))
    # margin aşırı olmasın diye güvenlik
    m = max(0, min(m, min(cell_h, cell_w) // 4))

    fen_rows = []
    for row in range(8):
        empty = 0
        fen_row = ""

        for col in range(8):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w

            yy1, yy2 = y1 + m, y2 - m
            xx1, xx2 = x1 + m, x2 - m

            # Margin çok geldiyse fallback
            if yy2 <= yy1 or xx2 <= xx1:
                yy1, yy2, xx1, xx2 = y1, y2, x1, x2

            cell = img[yy1:yy2, xx1:xx2]
            x = preprocess_cell(cell)

            preds = model.predict(x, verbose=0)[0]
            cls_name = idx_to_class[int(np.argmax(preds))]
            fen_char = CLASS_TO_FEN.get(cls_name)

            if fen_char is None:
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += fen_char

        if empty:
            fen_row += str(empty)

        fen_rows.append(fen_row)

    # SADECE board (w - - 0 1 YOK)
    return "/".join(fen_rows)

def insert_fen(conn: sqlite3.Connection, file_id: int, fen_board: str, image_blob: bytes) -> bool:
    # Sen DB'de kolonu fen_format yaptın:
    cur = conn.execute(
        "INSERT OR IGNORE INTO chess_fen (file_id, fen_format, image_blob) VALUES (?, ?, ?)",
        (file_id, fen_board, sqlite3.Binary(image_blob)),
    )
    return cur.rowcount == 1

def main():
    if not IMAGES_DIR.exists():
        raise RuntimeError(f"IMAGES_DIR yok: {IMAGES_DIR}")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"MODEL_PATH yok: {MODEL_PATH}")
    if not CLASS_INDICES_PATH.exists():
        raise RuntimeError(f"CLASS_INDICES_PATH yok: {CLASS_INDICES_PATH}")

    files = [p for p in IMAGES_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXT]
    if not files:
        raise RuntimeError(f"Görsel yok: {IMAGES_DIR}")

    model = load_model(str(MODEL_PATH))
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}

    conn = create_connection()
    create_tables(conn)
    conn.execute("PRAGMA foreign_keys = ON;")

    inserted, skipped = 0, 0
    ops_since_commit = 0

    try:
        for img_path in files:
            h = sha256_file(img_path)
            file_id = get_or_create_file_id(conn, img_path, h)

            fen_board = board_image_to_fen_board(img_path, model, idx_to_class)
            blob = img_path.read_bytes()

            if insert_fen(conn, file_id, fen_board, blob):
                inserted += 1
                print(f"[INSERT] {img_path.name} -> {fen_board}")
            else:
                skipped += 1
                print(f"[SKIP]   {img_path.name} (zaten var)")

            ops_since_commit += 1
            if ops_since_commit >= BATCH_COMMIT:
                conn.commit()
                print(f"[COMMIT] inserted={inserted} skipped={skipped}")
                ops_since_commit = 0

        conn.commit()

    except KeyboardInterrupt:
        conn.commit()
        print("\n[INTERRUPT] Son durum commit edildi.")
        print(f"inserted={inserted} skipped={skipped}")

    finally:
        conn.close()

    print(f"\nInserted: {inserted}")
    print(f"Skipped : {skipped}")

if __name__ == "__main__":
    main()
