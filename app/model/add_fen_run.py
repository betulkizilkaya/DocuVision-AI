#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
from app.core.paths import DB_PATH, ROOT_DIR

# --- AYARLAR ---
PIECE_MODEL_PATH = ROOT_DIR / "app" / "model" / "models" / "chess_model_finetuned.keras"
CLASS_INDICES_PATH = ROOT_DIR / "app" / "model" / "models" / "class_indices.json"

IMG_SIZE = (64, 64)

# Model çıktılarını FEN karakterlerine eşleyen sözlük
CLASS_TO_FEN = {
    "White_Pawn": "P", "White_Rook": "R", "White_Knight": "N", "White_Bishop": "B",
    "White_Queen": "Q", "White_King": "K",
    "Black_Pawn": "p", "Black_Rook": "r", "Black_Knight": "n", "Black_Bishop": "b",
    "Black_Queen": "q", "Black_King": "k",
    "Empty_Square": None,
}


def load_resources():
    print("[INIT] Model ve sınıf indeksleri yükleniyor...")
    model = load_model(str(PIECE_MODEL_PATH))
    with open(str(CLASS_INDICES_PATH), "r", encoding="utf-8") as f:
        idx_to_class = {int(v): k for k, v in json.load(f).items()}
    return model, idx_to_class


def predict_fen(board_img, model, idx_to_class):
    """Görseli 8x8 parçaya böler ve FEN satırlarını oluşturur."""
    h, w = board_img.shape[:2]
    sq_h, sq_w = h // 8, w // 8

    fen_rows = []
    for r in range(8):
        empty_count = 0
        current_row_fen = ""
        for c in range(8):
            # Kareyi kırp
            cell = board_img[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w]

            # Ön İşleme: Gri tonlama ve boyutlandırma
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            cell_res = cv2.resize(cell_gray, IMG_SIZE)
            cell_input = cell_res.astype("float32") / 255.0
            cell_input = np.expand_dims(cell_input, axis=(0, -1))

            # Model Tahmini
            preds = model.predict(cell_input, verbose=0)[0]
            class_idx = np.argmax(preds)
            class_name = idx_to_class[class_idx]
            fen_char = CLASS_TO_FEN.get(class_name)

            if fen_char is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    current_row_fen += str(empty_count)
                    empty_count = 0
                current_row_fen += fen_char

        if empty_count > 0:
            current_row_fen += str(empty_count)
        fen_rows.append(current_row_fen)

    return "/".join(fen_rows)


def main():
    model, idx_to_class = load_resources()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # 1. Yeni tabloyu oluştur (fen_format isminde)
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS fen_format
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     image_id
                     INTEGER,
                     board_index
                     INTEGER,
                     fen_text
                     TEXT,
                     UNIQUE
                 (
                     image_id,
                     board_index
                 )
                     )
                 """)

    # 2. İşlenecek tahtaları al
    rows = conn.execute("SELECT image_id, board_index, blob_png FROM final_boards").fetchall()
    print(f"[RUN] {len(rows)} adet tahta analiz ediliyor...")

    for row in rows:
        img_id, b_idx = row["image_id"], row["board_index"]

        # Blob'u resme çevir
        arr = np.frombuffer(row["blob_png"], np.uint8)
        board_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if board_img is None: continue

        try:
            # FEN Tahmini
            fen_str = predict_fen(board_img, model, idx_to_class)

            # 3. fen_format tablosuna kaydet
            conn.execute("""
                INSERT OR REPLACE INTO fen_format (image_id, board_index, fen_text)
                VALUES (?, ?, ?)
            """, (img_id, b_idx, fen_str))

            print(f"[OK] image_id={img_id} index={b_idx} -> {fen_str}")

        except Exception as e:
            print(f"[ERR] image_id={img_id} işlenirken hata oluştu: {e}")

        conn.commit()

    conn.close()
    print("[FINISH] İşlem tamamlandı. Sonuçlar 'fen_format' tablosuna kaydedildi.")


if __name__ == "__main__":
    main()