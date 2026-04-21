#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path

# Senin kendi fonksiyonların (FEN üretimi ve kayıt için)
from stage_fen import (
    FenParams,
    ensure_tables,
    load_piece_model_and_index,
    save_board_and_fen,
)
from app.core.paths import DB_PATH, ROOT_DIR

# --- AYARLAR ---
# Eğittiğin modeli buraya koyduğunu varsayıyorum
YOLO_MODEL_PATH = ROOT_DIR / "app" / "model" / "models" / "best.pt"
PIECE_MODEL_PATH = ROOT_DIR / "app" / "model" / "models" / "chess_model_v3.keras"
CLASS_INDICES_PATH = ROOT_DIR / "app" / "model" / "models" / "class_indices.json"

MIN_YOLO_CONF = 0.50  # YOLO'nun "bu kesin tahtadır" deme eşiği

SQL_PAGES = """
            SELECT pi.id, pi.blob
            FROM pdf_images pi
                     JOIN image_features f ON f.image_id = pi.id
            WHERE f.is_chessboard = 1 \
            """


def decode_page_blob(blob: bytes):
    arr = np.frombuffer(blob, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    # 1. Modelleri Yükle
    print("[INIT] Modeller yükleniyor...")
    yolo = YOLO(str(YOLO_MODEL_PATH))

    fen_p = FenParams(
        piece_model_path=str(PIECE_MODEL_PATH),
        class_indices_path=str(CLASS_INDICES_PATH)
    )
    piece_model, idx_to_class = load_piece_model_and_index(fen_p)

    # 2. Veritabanı Bağlantısı
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_tables(conn)

    # 3. Sayfaları Getir
    rows = conn.execute(SQL_PAGES).fetchall()
    print(f"[RUN] {len(rows)} sayfa YOLO ile işlenecek...")

    for row in rows:
        image_id = int(row["id"])
        page = decode_page_blob(row["blob"])
        if page is None: continue

        # 4. YOLO ile Tespit Yap
        # verbose=False yaparak terminalin gereksiz dolmasını engelliyoruz
        results = yolo.predict(source=page, conf=MIN_YOLO_CONF, verbose=False)

        # Her bir sayfa için bulunan tahtaları dön
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                # Koordinatları al
                x1, y1, x2, y2 = map(int, box)

                # Hafif bir pay (padding) ekleyelim ki taşlar tam çıksın (isteğe bağlı)
                pad = 2
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(page.shape[1], x2 + pad), min(page.shape[0], y2 + pad)

                # Tahtayı Kırp
                board_bgr = page[y1:y2, x1:x2]

                if board_bgr.size == 0: continue

                # 5. FEN Analizi ve Veritabanına Kaydet
                try:
                    save_board_and_fen(
                        conn=conn,
                        image_id=image_id,
                        board_index=i,
                        board_bgr=board_bgr,
                        source="yolov11_auto",
                        clf_score=float(scores[i]),
                        piece_model=piece_model,
                        idx_to_class=idx_to_class,
                    )
                    print(f"[OK] image_id={image_id} | Tahta {i} yakalandı (Score: {scores[i]:.2f})")
                except Exception as e:
                    print(f"[ERR] image_id={image_id} Tahta {i} kaydedilemedi: {e}")

        conn.commit()

    conn.close()
    print("[FINISH] İşlem başarıyla tamamlandı.")


if __name__ == "__main__":
    main()