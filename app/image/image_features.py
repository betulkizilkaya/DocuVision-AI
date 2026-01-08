import sqlite3
import io
import json
from collections import Counter
from typing import Optional, Tuple, List

from PIL import Image

from app.core.paths import DB_PATH


# Karelik toleransı
SQUARE_TOLERANCE = 0.9


def create_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(DB_PATH),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def fetch_images(conn: sqlite3.Connection) -> List[tuple[int, bytes]]:
    cur = conn.cursor()
    cur.execute("SELECT id, blob FROM pdf_images ORDER BY file_id, page_no, image_index")
    rows = cur.fetchall()
    return [(int(r["id"]), bytes(r["blob"])) for r in rows]


def feature_exists(conn: sqlite3.Connection, image_id: int) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM image_features WHERE image_id=? LIMIT 1", (image_id,))
    return cur.fetchone() is not None


# ---------------------------
# IMAGE UTILS
# ---------------------------
def is_grayscale(img: Image.Image) -> bool:
    """
    Görsel siyah-beyaz mı?
    RGB'ye çevirip örnek piksel kontrolü yapar.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    pixels = img.getdata()
    total = len(pixels)
    if total == 0:
        return True

    step = max(1, total // 5000)  # max ~5000 örnek
    for r, g, b in list(pixels)[::step]:
        if abs(r - g) > 3 or abs(g - b) > 3 or abs(r - b) > 3:
            return False
    return True


def get_top_colors(img: Image.Image, n: int = 5) -> list[tuple[tuple[int, int, int], float]]:
    """
    En sık geçen n rengi döndürür: [((r,g,b), oran), ...]
    Not: Büyük görsellerde Counter ağır olabilir; burada basitlik için bıraktım.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    pixels = list(img.getdata())
    if not pixels:
        return []

    # performans için örnekleme (istersen kapat)
    if len(pixels) > 300_000:
        step = max(1, len(pixels) // 100_000)  # ~100k örnek
        pixels = pixels[::step]

    count = Counter(pixels)
    most_common = count.most_common(n)
    total = len(pixels)
    return [(tuple(c), round(cnt / total, 4)) for c, cnt in most_common]


def analyze_image(image_bytes: bytes) -> Optional[Tuple[int, int, float, int, int, Optional[str]]]:
    """
    image_features şemasına uygun çıktı:
      width, height, aspect_ratio, is_square, is_grayscale, top_colors(JSON or NULL)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()

        width, height = img.size
        if width <= 0 or height <= 0:
            return None

        aspect_ratio = round(width / height, 6)

        ratio = min(width, height) / max(width, height)
        is_square = 1 if ratio >= SQUARE_TOLERANCE else 0

        gray = 1 if is_grayscale(img) else 0

        if gray:
            top_colors = None
        else:
            # JSON olarak sakla (db.py top_colors TEXT)
            top_colors = json.dumps(get_top_colors(img), ensure_ascii=False)

        return width, height, aspect_ratio, is_square, gray, top_colors

    except Exception:
        return None


def insert_features(conn: sqlite3.Connection, image_id: int, features: Tuple) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO image_features
        (image_id, width, height, aspect_ratio, is_square, is_grayscale, top_colors)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (image_id, *features),
    )


# ---------------------------
# MAIN
# ---------------------------
def process_all(*, skip_existing: bool = True) -> None:
    conn = create_connection()
    try:
        images = fetch_images(conn)
        print(f"[INFO] {len(images)} görsel bulundu. Özellikler hesaplanıyor...")

        inserted = 0
        skipped = 0
        failed = 0

        cur = conn.cursor()

        for idx, (img_id, blob) in enumerate(images, start=1):

            # ---- PROGRESS PRINT (HER 100 GÖRSELDE BİR) ----
            if idx % 100 == 0:
                print(f"[PROGRESS] {idx}/{len(images)} görsel işlendi")

            if skip_existing and feature_exists(conn, img_id):
                skipped += 1
                continue

            features = analyze_image(blob)
            if features is None:
                failed += 1
                continue

            insert_features(conn, img_id, features)
            inserted += 1

            if inserted % 250 == 0:
                conn.commit()

        conn.commit()
        print(f"[✓] Bitti. inserted={inserted} skipped={skipped} failed={failed}")

    finally:
        conn.close()


if __name__ == "__main__":
    process_all(skip_existing=True)
