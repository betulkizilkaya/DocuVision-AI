import sqlite3
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"


def clear_ocr_table():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # Sadece OCR tablosunun içindeki verileri siler (Tablo yapısı kalır)
        cur.execute("DELETE FROM ocr_extracts")
        conn.commit()

        count = cur.rowcount
        print(f"[✓] Temizlik tamamlandı. {count} adet hatalı/boş kayıt silindi.")

        conn.close()
    except Exception as e:
        print(f"Hata: {e}")


if __name__ == "__main__":
    clear_ocr_table()