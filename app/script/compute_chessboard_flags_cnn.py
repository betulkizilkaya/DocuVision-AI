import io
import sqlite3
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from app.core.paths import DB_PATH, ROOT_DIR
from app.core.db import create_connection


MODEL_PATH = ROOT_DIR / "data" / "models" / "chessboard_cnn_v1.keras"
THRESHOLD = 0.65
IMG_SIZE = (128, 128)


class ChessboardCNNClassifier:
    def __init__(self, model_path: Path):
        print(f"[MODEL] CNN model yükleniyor: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        img = pil_image.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, blob_data: bytes):
        img = Image.open(io.BytesIO(blob_data))
        x = self._preprocess(img)

        score = float(self.model.predict(x, verbose=0)[0][0])
        is_chessboard = 1 if score >= THRESHOLD else 0
        return is_chessboard, score


def get_pending_images(conn: sqlite3.Connection):
    cur = conn.cursor()
    query = """
        SELECT f.image_id, p.blob
        FROM image_features f
        INNER JOIN pdf_images p ON p.id = f.image_id
        WHERE f.is_chessboard IS NULL
    """
    cur.execute(query)
    return cur.fetchall()


def update_chessboard_flag(
    conn: sqlite3.Connection,
    image_id: int,
    is_chessboard: int,
    score: float,
):
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE image_features
        SET is_chessboard = ?, chessboard_score = ?
        WHERE image_id = ?
        """,
        (is_chessboard, score, image_id),
    )
    conn.commit()


def run_chessboard_detection():
    print("--- [GÖREV: CNN SATRANÇ TAHTASI TESPİTİ BAŞLIYOR] ---")
    print(f"[DB] {DB_PATH}")
    print(f"[CFG] THRESHOLD = {THRESHOLD}")

    conn = create_connection()

    try:
        classifier = ChessboardCNNClassifier(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"❌ HATA: {e}")
        conn.close()
        return

    images = get_pending_images(conn)
    print(f"[INFO] İşlenecek {len(images)} görsel bulundu.")

    detected = 0
    failed = 0

    for idx, (img_id, blob) in enumerate(images, start=1):
        try:
            is_chessboard, score = classifier.predict(blob)
            update_chessboard_flag(conn, img_id, is_chessboard, score)

            if is_chessboard == 1:
                detected += 1
                print(f"   → ID {img_id}: TAHTA TESPİT EDİLDİ | score={score:.4f}")

            if idx % 100 == 0:
                print(f"[PROGRESS] {idx}/{len(images)} işlendi")

        except Exception as e:
            failed += 1
            print(f"[WARN] image_id={img_id} işlenemedi: {e}")

    conn.close()
    print(f"[OK] Bitti | detected={detected} | failed={failed}")
    print("--- [GÖREV: CNN SATRANÇ TAHTASI TESPİTİ TAMAMLANDI] ---")


if __name__ == "__main__":
    run_chessboard_detection()