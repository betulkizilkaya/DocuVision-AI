import fitz
import sqlite3
import hashlib
import io
from pathlib import Path
from PIL import Image

from app.core.paths import DB_PATH, ROOT_DIR, DATA_DIR

PDF_DIR = DATA_DIR

# Thumbnail’lerin kaydedileceği klasör
OUT_DIR = ROOT_DIR / "temp" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_connection(db_file=DB_PATH):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_file))


def safe_clip_rect(page: fitz.Page, rect: fitz.Rect):
    if rect is None:
        return None

    coords = [rect.x0, rect.y0, rect.x1, rect.y1]
    if any(c != c for c in coords):  # NaN
        return None

    r = fitz.Rect(rect).normalize()
    r = r & page.rect

    if r.is_empty or r.width <= 1 or r.height <= 1:
        return None

    return r


def extract_images_from_pdf(pdf_path: Path, conn):
    doc = fitz.open(pdf_path)
    cursor = conn.cursor()

    with open(pdf_path, "rb") as f:
        file_hash = sha256_bytes(f.read())

    cursor.execute(
        "INSERT OR IGNORE INTO file_index (filename, sha256) VALUES (?, ?)",
        (pdf_path.name, file_hash)
    )
    conn.commit()

    cursor.execute("SELECT id FROM file_index WHERE filename = ?", (pdf_path.name,))
    row = cursor.fetchone()
    if not row:
        return
    file_id = row[0]

    print(f"[+] {pdf_path.name} işleniyor...")

    MAX_PIXELS = 40_000_000
    image_counter = 0

    for page_no, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]

            try:
                raw_bytes = doc.extract_image(xref)["image"]
            except Exception:
                continue
            sha_raw = sha256_bytes(raw_bytes)

            rects = page.get_image_rects(xref)
            if not rects:
                continue

            for rect_i, rect in enumerate(rects):
                safe_rect = safe_clip_rect(page, rect)
                if safe_rect is None:
                    continue

                render_bytes = None
                for scale in (2, 1):
                    try:
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(scale, scale),
                            clip=safe_rect,
                            alpha=False
                        )
                        if pix.width * pix.height > MAX_PIXELS:
                            continue
                        render_bytes = pix.tobytes("png")
                        break
                    except Exception:
                        continue

                if render_bytes is None:
                    continue

                sha_render = sha256_bytes(render_bytes)
                image_index_db = img_index * 1000 + rect_i

                cursor.execute("""
                    INSERT OR IGNORE INTO pdf_images
                    (file_id, page_no, image_index, xref, rect_i, sha256, sha256_raw, blob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_id,
                    page_no,
                    image_index_db,
                    xref,
                    rect_i,
                    sha_render,
                    sha_raw,
                    sqlite3.Binary(render_bytes)
                ))

                try:
                    image = Image.open(io.BytesIO(render_bytes))
                    image.thumbnail((128, 128))
                    image.save(OUT_DIR / f"{pdf_path.stem}_p{page_no}_{img_index}_{rect_i}.png")
                except Exception:
                    pass

                image_counter += 1

    conn.commit()
    print(f"[✓] {pdf_path.name}: {image_counter} görsel")


def process_all_pdfs():
    conn = create_connection(DB_PATH)
    try:
        for pdf_file in PDF_DIR.glob("*.pdf"):
            extract_images_from_pdf(pdf_file, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    process_all_pdfs()
