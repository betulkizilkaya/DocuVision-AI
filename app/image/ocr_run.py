import sqlite3
import io
import os
from pathlib import Path
from PIL import Image

import ocr_engine

CURRENT_FILE = Path(__file__).resolve()
IMAGE_DIR = CURRENT_FILE.parent
ROOT_DIR = IMAGE_DIR.parent.parent
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"


def create_connection():
    if not os.path.exists(DB_PATH.parent):
        os.makedirs(DB_PATH.parent)
    return sqlite3.connect(DB_PATH)


def get_pending_images(conn):
    """
    Daha önce 'ocr_extracts' tablosuna HİÇ eklenmemiş görselleri getir.
    """
    cur = conn.cursor()
    query = """
            SELECT T1.id, T1.blob
            FROM pdf_images T1
                     LEFT JOIN ocr_extracts T2 ON T1.id = T2.image_id
            WHERE T2.id IS NULL \
            """
    cur.execute(query)
    return cur.fetchall()


def save_result(conn, img_id, text):
    """
    Sonucu veritabanına yazar.
    """
    cur = conn.cursor()
    cur.execute("""
                INSERT INTO ocr_extracts (image_id, text_raw)
                VALUES (?, ?)
                """, (img_id, text))
    conn.commit()


def process():
    conn = create_connection()
    images = get_pending_images(conn)

    print(f"[INFO] İşlenecek {len(images)} yeni görsel bulundu.")
    print("-" * 50)
    print(f"{'ID':<6} | {'DURUM':<35}")
    print("-" * 50)

    success_count = 0

    for img_id, blob in images:
        try:
            # Görseli aç
            img = Image.open(io.BytesIO(blob))

            # Motora gönder (Tek fonksiyon)
            text_result = ocr_engine.run_ocr(img)

            # Kaydet
            save_result(conn, img_id, text_result)

            # Ekrana Bilgi Bas
            if text_result:
                # Metnin ilk 30 karakterini göster
                preview = text_result[:30].replace('\n', ' ') + "..."
                print(f"{img_id:<6} | {preview:<35}")
                success_count += 1
            else:
                # Gürültü veya boş ise
                # print(f"{img_id:<6} | [Boş/Gürültü - Kaydedilmedi]")
                pass  # Konsolu kirletmemek için boşları yazmayabiliriz

        except Exception as e:
            print(f"Hata (ID: {img_id}): {e}")

    conn.close()
    print("-" * 50)
    print(f"[✓] İşlem tamamlandı. {success_count} görselden anlamlı metin çıkarıldı.")


if __name__ == "__main__":
    process()