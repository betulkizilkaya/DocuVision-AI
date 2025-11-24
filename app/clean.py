import sqlite3
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"


def clean_database():
    if not DB_PATH.exists():
        print(f"[!] Veritabanı bulunamadı: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        print("Tablo siliniyor...")
        # Tabloyu tamamen uçuruyoruz ki db.py çalışınca yeni sütunlarla temiz kurulum yapsın.
        cursor.execute("DROP TABLE IF EXISTS image_similarity")
        conn.commit()
        print("[✓] 'image_similarity' tablosu başarıyla silindi.")
        print("    -> Şimdi 'app/db.py' dosyasını çalıştırarak tabloyu yeniden oluşturun.")
    except Exception as e:
        print(f"[!] Hata oluştu: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    clean_database()