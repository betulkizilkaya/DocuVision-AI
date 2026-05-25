#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import cv2
import numpy as np
import os
from pathlib import Path
from app.core.paths import DB_PATH, ROOT_DIR

# --- AYARLAR ---
OUTPUT_DIR = ROOT_DIR / "app" / "data" / "extracted_squares"
SQUARE_SIZE = (64, 64)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("SELECT image_id, board_index, blob_png FROM final_boards").fetchall()
    print(f"[RUN] {len(rows)} tahta parçalanıyor...")

    for row in rows:
        image_id = row["image_id"]
        board_idx = row["board_index"]

        arr = np.frombuffer(row["blob_png"], np.uint8)
        board_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if board_img is None: continue

        h, w = board_img.shape[:2]

        # Gönderdiğin görsele göre: Kırpma yapmıyoruz,
        # sadece tam bölünebilmesi için çok küçük bir ayarlama yapıyoruz.
        square_h = h // 8
        square_w = w // 8

        board_folder = OUTPUT_DIR / f"img_{image_id}_board_{board_idx}"
        os.makedirs(board_folder, exist_ok=True)

        for r in range(8):
            for c in range(8):
                # Koordinatlar
                y1, y2 = r * square_h, (r + 1) * square_h
                x1, x2 = c * square_w, (c + 1) * square_w

                square = board_img[y1:y2, x1:x2]

                if square.size == 0: continue

                # --- OPSİYONEL: Gürültü Azaltma ---
                # Görseldeki o çizgili dokuyu (halftone) hafifletmek için
                # küçük bir Gaussian Blur eklenebilir.
                # Bu, modelin dokuya değil şekle odaklanmasına yardımcı olur.
                # square = cv2.GaussianBlur(square, (3, 3), 0)

                # Boyutlandırma
                square_resized = cv2.resize(square, SQUARE_SIZE, interpolation=cv2.INTER_AREA)

                # Kaydet
                file_path = board_folder / f"sq_{r}_{c}.png"
                cv2.imwrite(str(file_path), square_resized)

        print(f"[OK] {board_folder.name} klasörüne 64 kare çıkarıldı.")

    conn.close()
    print("\n[FINISH] İşlem tamam. Bu karelerle modeli eğitebilirsin.")


if __name__ == "__main__":
    main()