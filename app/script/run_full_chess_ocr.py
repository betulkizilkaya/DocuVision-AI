import io
import os
import sqlite3
import tempfile

from PIL import Image

from app.core.paths import DB_PATH
from app.image.chess_notation_ocr import process_single_image


def main():
    success_count = 0
    error_count = 0

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    try:
        rows = cur.execute("""
            SELECT p.id, p.blob
            FROM pdf_images p
            INNER JOIN image_features f ON f.image_id = p.id
            WHERE f.is_chessboard = 0
            ORDER BY p.id
        """).fetchall()

        print(f"{len(rows)} görsel bulundu, OCR başlıyor...")

        for image_id, blob in rows:
            temp_path = None

            try:
                img = Image.open(io.BytesIO(blob)).convert("RGB")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    img.save(tmp.name)
                    temp_path = tmp.name

                process_single_image(image_id, temp_path)

                print(f"[OK] image_id={image_id}")
                success_count += 1

            except Exception as e:
                print(f"[ERROR] image_id={image_id}: {e}")
                error_count += 1

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    finally:
        conn.close()

    print("🔥 TÜM OCR TAMAMLANDI")
    print(f"Başarılı: {success_count}")
    print(f"Hatalı: {error_count}")


if __name__ == "__main__":
    main()