# add_fen_to_db.py
import io
import json
import sqlite3
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from app.core.paths import DB_PATH, ROOT_DIR


# ---------------------------
# MODEL PATHS
# ---------------------------
MODEL_PATH = ROOT_DIR / "app" / "model" / "models" / "chess_model_v3.keras"
CLASS_INDICES_PATH = ROOT_DIR / "app" / "model" / "models" / "class_indices.json"

# ---------------------------
# SETTINGS
# ---------------------------
IMG_SIZE = (64, 64)
UPSCALE = 2.5
CELL_MARGIN_RATIO = 0.08
USE_EQUALIZE = True  # (Bu flag artık preprocess'te kullanılmıyor; istersen kaldırabilirsin)

# ✅ İSTEDİĞİN FİLTRE: SADECE BU file_id'ler
FILTER_FILE_IDS = (2, 5, 11, 13, 14)

# Performans / çalışma kontrolü
BATCH_COMMIT = 50
LIMIT: Optional[int] = None          # None=hepsi, denemek için 200 gibi
MIN_CHESSBOARD_SCORE = 0.0           # istersen 0.35 gibi yap

CLASS_TO_FEN = {
    "White_Pawn": "P", "White_Rook": "R", "White_Knight": "N", "White_Bishop": "B",
    "White_Queen": "Q", "White_King": "K",
    "Black_Pawn": "p", "Black_Rook": "r", "Black_Knight": "n", "Black_Bishop": "b",
    "Black_Queen": "q", "Black_King": "k",
    "Empty_Square": None,
}


# ---------------------------
# DB
# ---------------------------
def create_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_chess_fen_table(conn: sqlite3.Connection) -> None:
    """
    chess_fen image_id bazlı ve blob'suz.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chess_fen(
            image_id INTEGER PRIMARY KEY,
            fen_format TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (image_id) REFERENCES pdf_images(id)
        );
        """
    )
    conn.commit()


def fetch_chessboard_images(conn: sqlite3.Connection) -> List[Tuple[int, bytes, float]]:
    """
    is_chessboard=1 olan görselleri, score + file_id filtresi ile getirir.
    """
    if not FILTER_FILE_IDS:
        raise ValueError("FILTER_FILE_IDS boş olamaz.")

    placeholders = ",".join(["?"] * len(FILTER_FILE_IDS))

    q = f"""
    SELECT
      pi.id AS image_id,
      pi.blob AS blob,
      COALESCE(f.chessboard_score, 0.0) AS chessboard_score
    FROM pdf_images pi
    JOIN image_features f ON f.image_id = pi.id
    WHERE f.is_chessboard = 1
      AND COALESCE(f.chessboard_score, 0.0) >= ?
      AND pi.file_id IN ({placeholders})
    ORDER BY chessboard_score DESC, pi.file_id, pi.page_no, pi.image_index
    """

    params = [float(MIN_CHESSBOARD_SCORE), *FILTER_FILE_IDS]

    if LIMIT is not None:
        q += " LIMIT ?"
        params.append(int(LIMIT))

    cur = conn.cursor()
    cur.execute(q, params)
    rows = cur.fetchall()

    return [
        (int(r["image_id"]), bytes(r["blob"]), float(r["chessboard_score"]))
        for r in rows
    ]


def fen_exists(conn: sqlite3.Connection, image_id: int) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM chess_fen WHERE image_id=? LIMIT 1", (image_id,))
    return cur.fetchone() is not None


def insert_fen(conn: sqlite3.Connection, image_id: int, fen_board: str) -> bool:
    cur = conn.execute(
        "INSERT OR IGNORE INTO chess_fen (image_id, fen_format) VALUES (?, ?)",
        (image_id, fen_board),
    )
    return cur.rowcount == 1


# ---------------------------
# IMAGE STANDARDIZATION
# ---------------------------
def blob_to_bgr_standardized(blob: bytes) -> np.ndarray:
    """
    DB blob'unu standardize eder:
    - PIL ile aç
    - RGB'ye çevir
    - PNG olarak yeniden encode et
    - OpenCV imdecode ile BGR al
    """
    img = Image.open(io.BytesIO(blob))
    img.load()

    if img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)

    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("imdecode failed after standardization")
    return bgr


def preprocess_cell(cell_bgr: np.ndarray) -> np.ndarray:
    """
    Eğitim pipeline'ına uyumlu preprocess:
    - grayscale
    - CLAHE ile kontrast stabilize
    - resize (küçültmede INTER_AREA)
    - normalize 0..1
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)

    x = gray.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=(0, -1))  # (1, 64, 64, 1)


def board_blob_to_fen_board(blob: bytes, model, idx_to_class: dict) -> str:
    img = blob_to_bgr_standardized(blob)

    # upscale
    img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    h, w, _ = img.shape
    cell_h = h // 8
    cell_w = w // 8
    if cell_h <= 0 or cell_w <= 0:
        raise RuntimeError(f"Invalid board size after resize: h={h}, w={w}")

    m = int(CELL_MARGIN_RATIO * min(cell_h, cell_w))
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

    return "/".join(fen_rows)


# ---------------------------
# MAIN
# ---------------------------
def main() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"MODEL_PATH yok: {MODEL_PATH}")
    if not CLASS_INDICES_PATH.exists():
        raise RuntimeError(f"CLASS_INDICES_PATH yok: {CLASS_INDICES_PATH}")

    model = load_model(str(MODEL_PATH))
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}

    conn = create_connection()
    try:
        ensure_chess_fen_table(conn)

        rows = fetch_chessboard_images(conn)
        print(
            f"[INFO] file_id IN {FILTER_FILE_IDS} & chessboard=1: {len(rows)} görsel bulundu "
            f"(MIN_SCORE={MIN_CHESSBOARD_SCORE})."
        )

        inserted, skipped, failed = 0, 0, 0
        ops = 0

        for image_id, blob, score in rows:
            if fen_exists(conn, image_id):
                skipped += 1
                continue

            try:
                fen_board = board_blob_to_fen_board(blob, model, idx_to_class)
                ok = insert_fen(conn, image_id, fen_board)
                if ok:
                    inserted += 1
                    print(f"[INSERT] image_id={image_id} score={score:.2f} -> {fen_board}")
                else:
                    skipped += 1

            except Exception as e:
                failed += 1
                print(f"[FAIL] image_id={image_id} score={score:.2f}: {e}")

            ops += 1
            if ops >= BATCH_COMMIT:
                conn.commit()
                print(f"[COMMIT] inserted={inserted} skipped={skipped} failed={failed}")
                ops = 0

        conn.commit()
        print(f"[OK] Done. inserted={inserted} skipped={skipped} failed={failed}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
