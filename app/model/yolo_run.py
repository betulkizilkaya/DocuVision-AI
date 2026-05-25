import sqlite3
import numpy as np
import cv2
import time
from ultralytics import YOLO
from app.core.paths import DB_PATH, ROOT_DIR

# --- AYARLAR ---
YOLO_MODEL_PATH = ROOT_DIR / "app" / "model" / "models" / "best.pt"
MIN_YOLO_CONF = 0.50

SQL_PAGES = """
            SELECT pi.id, pi.blob
            FROM pdf_images pi
                     JOIN image_features f ON f.image_id = pi.id
            WHERE f.is_chessboard = 1 
            """

def ensure_tables(conn: sqlite3.Connection):
    # final_boards tablosunun varlığından emin oluyoruz
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS final_boards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            board_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            clf_score REAL,
            w INTEGER NOT NULL,
            h INTEGER NOT NULL,
            blob_png BLOB NOT NULL,
            created_at INTEGER NOT NULL,
            UNIQUE(image_id, board_index)
        )
        """
    )+
    conn.commit()

def decode_page_blob(blob: bytes):
    arr = np.frombuffer(blob, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def main():
    print("[INIT] YOLO modeli yükleniyor...")
    yolo = YOLO(str(YOLO_MODEL_PATH))

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_tables(conn)

    rows = conn.execute(SQL_PAGES).fetchall()
    print(f"[RUN] {len(rows)} sayfa taranıyor...")

    for row in rows:
        image_id = int(row["id"])
        page = decode_page_blob(row["blob"])
        if page is None: continue

        # YOLO ile tespit
        results = yolo.predict(source=page, conf=MIN_YOLO_CONF, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)

                # Padding (Taşların tam görünmesi için 5 piksel daha sağlıklı olabilir)
                pad = 5
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(page.shape[1], x2 + pad), min(page.shape[0], y2 + pad)

                # Tahtayı Kırp
                board_img = page[y1:y2, x1:x2]
                if board_img.size == 0: continue

                # Boyutları al
                height, width = board_img.shape[:2]

                try:
                    # Görseli PNG formatında blob'a çevir
                    _, buffer = cv2.imencode(".png", board_img)
                    board_blob = buffer.tobytes()

                    # final_boards tablosuna kayıt
                    conn.execute("""
                        INSERT OR REPLACE INTO final_boards 
                        (image_id, board_index, source, clf_score, w, h, blob_png, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        image_id,
                        i,
                        "yolov11_auto",
                        float(scores[i]),
                        width,
                        height,
                        board_blob,
                        int(time.time())
                    ))

                    print(f"[OK] image_id={image_id} | Tahta {i} ({width}x{height}) kaydedildi.")
                except Exception as e:
                    print(f"[ERR] image_id={image_id} Kayıt hatası: {e}")

        conn.commit()

    conn.close()
    print("[FINISH] İşlem tamamlandı. Tüm tahtalar final_boards tablosuna aktarıldı.")

if __name__ == "__main__":
    main()