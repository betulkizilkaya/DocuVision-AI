import sqlite3
import io
from typing import Optional, List, Tuple

from PIL import Image

from app.core.paths import DB_PATH
from app.image import ocr_engine


# ---------------------------
# DB
# ---------------------------
def create_connection() -> sqlite3.Connection:
    """
    db.py ile aynı bağlantı ayarları.
    """
    conn = sqlite3.connect(
        str(DB_PATH),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def get_pending_images(conn: sqlite3.Connection) -> List[tuple[int, bytes]]:
    """
    Daha önce ocr_extracts'e hiç yazılmamış (image_id yok) görselleri getir.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.id AS image_id, p.blob AS blob
        FROM pdf_images p
        LEFT JOIN ocr_extracts o ON p.id = o.image_id
        WHERE o.image_id IS NULL
        ORDER BY p.file_id, p.page_no, p.image_index
        """
    )
    rows = cur.fetchall()
    return [(int(r["image_id"]), bytes(r["blob"])) for r in rows]


def save_result(conn: sqlite3.Connection, image_id: int, text: Optional[str]) -> None:
    """
    Sonucu ocr_extracts tablosuna yazar.

    db.py şeması:
      ocr_extracts(image_id INTEGER UNIQUE, text_raw TEXT, FOREIGN KEY(image_id) REFERENCES pdf_images(id))
    """
    cur = conn.cursor()

    # image_id UNIQUE olduğu için tekrar çalıştırmalarda hata vermesin
    cur.execute(
        """
        INSERT OR REPLACE INTO ocr_extracts (image_id, text_raw)
        VALUES (?, ?)
        """,
        (image_id, text),
    )
    conn.commit()


# ---------------------------
# MAIN
# ---------------------------
def process(*, skip_null_writes: bool = False) -> None:
    """
    skip_null_writes=False:
      - OCR sonucu None olsa bile DB'ye NULL olarak yazar (idempotent, tekrar tarama yapmaz)
    skip_null_writes=True:
      - None ise DB'ye hiç yazmaz (sonra tekrar pending olarak gelir)
    """
    conn = create_connection()
    try:
        images = get_pending_images(conn)

        print(f"[INFO] İşlenecek {len(images)} yeni görsel bulundu.")
        print("-" * 70)
        print(f"{'ID':<8} | {'DURUM':<55}")
        print("-" * 70)

        meaningful_count = 0

        for image_id, blob in images:
            try:
                img = Image.open(io.BytesIO(blob))
                img.load()

                text_result = ocr_engine.run_ocr(img)  # None veya string

                if skip_null_writes and not text_result:
                    # boş/gürültüyü DB'ye yazma; tekrar pending kalır
                    continue

                save_result(conn, image_id, text_result)

                if text_result:
                    preview = text_result.replace("\n", " ")[:55]
                    if len(text_result) > 55:
                        preview += "..."
                    print(f"{image_id:<8} | {preview:<55}")
                    meaningful_count += 1

            except Exception as e:
                # Hata durumunda da istersen DB'ye NULL yazıp "pending"den düşürebilirsin.
                # Şimdilik sadece logluyoruz.
                print(f"[ERROR] ID={image_id}: {e}")

        print("-" * 70)
        print(f"[✓] İşlem tamamlandı. {meaningful_count} görselden anlamlı metin çıkarıldı.")

    finally:
        conn.close()


if __name__ == "__main__":
    process(skip_null_writes=False)
