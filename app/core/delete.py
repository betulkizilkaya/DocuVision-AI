import sqlite3
from app.core.paths import DB_PATH, ROOT_DIR


# ---------------------------
# DB PATHS
# ---------------------------
SRC_DB = ROOT_DIR / "db_eski4" / "corpus.sqlite"
DST_DB = ROOT_DIR / "db" / "corpus.sqlite"


def copy_tables():
    if not SRC_DB.exists():
        raise FileNotFoundError(f"Kaynak DB bulunamadı: {SRC_DB}")

    if not DST_DB.exists():
        raise FileNotFoundError(f"Hedef DB bulunamadı: {DST_DB}")

    conn = sqlite3.connect(DST_DB)
    cur = conn.cursor()

    # Foreign key geçici kapat
    cur.execute("PRAGMA foreign_keys = OFF;")

    # Kaynak DB attach
    cur.execute(f"ATTACH DATABASE '{SRC_DB.as_posix()}' AS src;")

    try:
        conn.execute("BEGIN;")

        # -------------------------
        # file_index
        # -------------------------
        cur.execute("""
            INSERT INTO file_index (id, filename, path, sha256)
            SELECT id, filename, path, sha256
            FROM src.file_index;
        """)

        # -------------------------
        # pdf_images
        # -------------------------
        cur.execute("""
            INSERT INTO pdf_images
            (id, file_id, page_no, image_index,
             xref, rect_i,
             sha256, sha256_raw, blob)
            SELECT
             id, file_id, page_no, image_index,
             xref, rect_i,
             sha256, sha256_raw, blob
            FROM src.pdf_images;
        """)

        conn.commit()
        print("✅ db_eski4 → db corpus.sqlite aktarımı tamamlandı")

    except Exception as e:
        conn.rollback()
        raise

    finally:
        cur.execute("DETACH DATABASE src;")
        cur.execute("PRAGMA foreign_keys = ON;")
        conn.close()


if __name__ == "__main__":
    copy_tables()
