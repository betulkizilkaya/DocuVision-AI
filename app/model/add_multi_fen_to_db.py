import io
import json
import sqlite3
import cv2
import numpy as np
import os
from pathlib import Path
from tensorflow.keras.models import load_model
from app.core.paths import DB_PATH, ROOT_DIR

# --- AYARLAR ---
MODEL_PATH = ROOT_DIR / "app" / "model" / "models" / "chess_model_v3.keras"
CLASS_INDICES_PATH = ROOT_DIR / "app" / "model" / "models" / "class_indices.json"
DEBUG_DIR = ROOT_DIR / "debug_boards"  # Tahtaların kaydedileceği yer
IMG_SIZE = (64, 64)
MIN_CHESSBOARD_SCORE = 0.35

# Klasörü oluştur
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_TO_FEN = {
    "White_Pawn": "P", "White_Rook": "R", "White_Knight": "N", "White_Bishop": "B",
    "White_Queen": "Q", "White_King": "K",
    "Black_Pawn": "p", "Black_Rook": "r", "Black_Knight": "n", "Black_Bishop": "b",
    "Black_Queen": "q", "Black_King": "k",
    "Empty_Square": None,
}


def preprocess_cell(cell_bgr):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray, IMG_SIZE)
    x = res.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=(0, -1))


def detect_boards_in_blob(blob: bytes) -> list:
    nparr = np.frombuffer(blob, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None: return []
    hierarchy = hierarchy[0]

    board_crops = []
    h_img, w_img = img.shape[:2]

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < (h_img * w_img * 0.01): continue

        child_count = 0
        child_idx = hierarchy[i][2]
        while child_idx != -1:
            child_count += 1
            child_idx = hierarchy[child_idx][0]

        # 64 kare kuralı
        if 50 < child_count < 100:
            x, y, w, h = cv2.boundingRect(cnt)
            # %5 Padding (Taşları kurtarmak için)
            p = int(w * 0.05)
            crop = img[max(0, y - p):min(h_img, y + h + p), max(0, x - p):min(w_img, x + w + p)]
            board_crops.append(crop)

    return board_crops


def board_img_to_fen(board_img, model, idx_to_class):
    h_orig, w_orig = board_img.shape[:2]
    p = int(w_orig * 0.05)
    clean = board_img[p:h_orig - p, p:w_orig - p]
    clean = cv2.resize(clean, (512, 512))

    cell_size = 512 // 8
    fen_rows = []

    for r in range(8):
        empty, fen_row = 0, ""
        for c in range(8):
            cell = clean[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size]
            x_input = preprocess_cell(cell)
            preds = model.predict(x_input, verbose=0)[0]
            cls_name = idx_to_class[int(np.argmax(preds))]
            char = CLASS_TO_FEN.get(cls_name)

            if char is None:
                empty += 1
            else:
                if empty: fen_row += str(empty); empty = 0
                fen_row += char
        if empty: fen_row += str(empty)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


def main():
    model = load_model(str(MODEL_PATH))
    with open(CLASS_INDICES_PATH, "r") as f:
        idx_to_class = {int(v): k for k, v in json.load(f).items()}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    conn.execute(
        "CREATE TABLE IF NOT EXISTS chess_fen_multi (id INTEGER PRIMARY KEY, image_id INTEGER, board_index INTEGER, fen_format TEXT, UNIQUE(image_id, board_index))")

    query = "SELECT pi.id, pi.blob FROM pdf_images pi JOIN image_features f ON f.image_id = pi.id WHERE f.is_chessboard = 1 AND f.chessboard_score >= ?"
    rows = conn.execute(query, (MIN_CHESSBOARD_SCORE,)).fetchall()

    print(f"[BAŞLADI] {len(rows)} sayfa taranıyor...")

    for row in rows:
        image_id = row['id']
        boards = detect_boards_in_blob(row['blob'])

        for i, b_img in enumerate(boards):
            # --- KAYIT SATIRI BURASI ---
            save_path = DEBUG_DIR / f"id_{image_id}_board_{i}.png"
            cv2.imwrite(str(save_path), b_img)

            try:
                fen = board_img_to_fen(b_img, model, idx_to_class)
                conn.execute(
                    "INSERT OR REPLACE INTO chess_fen_multi (image_id, board_index, fen_format) VALUES (?, ?, ?)",
                    (image_id, i, fen))
                print(f"ID {image_id} - Tahta {i} kaydedildi ve FEN üretildi.")
            except Exception as e:
                print(f"Hata ID {image_id}: {e}")
        conn.commit()

    conn.close()
    print(f"[TAMAMLANDI] Görseller '{DEBUG_DIR}' klasörüne bakabilirsin.")


if __name__ == "__main__":
    main()