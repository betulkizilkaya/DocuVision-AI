import sqlite3
from app.core.paths import DB_PATH


def delete_multi_fen_table():
    try:
        # Veritabanına bağlan
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Tabloyu silme komutu
        cursor.execute("DROP TABLE IF EXISTS chess_fen_multi;")

        conn.commit()
        print("[BAŞARILI] 'chess_fen_multi' tablosu tamamen silindi.")

    except Exception as e:
        print(f"[HATA] Tablo silinirken bir sorun oluştu: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    delete_multi_fen_table()
