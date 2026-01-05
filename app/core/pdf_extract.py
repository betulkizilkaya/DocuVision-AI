import fitz          # PyMuPDF
import sqlite3
import hashlib
import io
from pathlib import Path
from PIL import Image

# Bu dosyanın bulunduğu klasör (örnek: app/)
APP_DIR = Path(__file__).resolve().parent

# Proje kökü = app'in bir üstü
ROOT_DIR = APP_DIR.parent.parent

# Veritabanı yolu (app'in dışında, kökte)
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"

# PDF dosyalarının bulunduğu klasör (app'in dışında, kökte)
PDF_DIR = ROOT_DIR / "data"

# Thumbnail’lerin kaydedileceği klasör (app'in dışında, kökte)
OUT_DIR = ROOT_DIR / "temp" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    """Verilen verinin SHA-256 hash'ini döndürür."""
    return hashlib.sha256(data).hexdigest()


def create_connection(db_file=DB_PATH):
    return sqlite3.connect(db_file)


def safe_clip_rect(page: fitz.Page, rect: fitz.Rect) -> fitz.Rect | None:
    """
    Rect'i normalize eder, sayfa sınırlarına kırpar.
    Geçersiz/boş rect ise None döner.
    """
    if rect is None:
        return None

    # NaN kontrolü
    coords = [rect.x0, rect.y0, rect.x1, rect.y1]
    if any(c != c for c in coords):  # NaN
        return None

    r = fitz.Rect(rect).normalize()

    # Sayfa sınırına kırp (intersection)
    r = r & page.rect

    if r.is_empty or r.width <= 1 or r.height <= 1:
        return None

    return r


def extract_images_from_pdf(pdf_path: Path, conn):
    """PDF içindeki tüm görselleri çıkarır ve pdf_images tablosuna kaydeder."""
    doc = fitz.open(pdf_path)
    cursor = conn.cursor()

    # file_index tablosunda dosyayı kaydet
    with open(pdf_path, "rb") as f:
        file_hash = sha256_bytes(f.read())

    cursor.execute(
        "INSERT OR IGNORE INTO file_index (filename, sha256) VALUES (?, ?)",
        (pdf_path.name, file_hash)
    )
    conn.commit()

    # file_id'yi al
    cursor.execute("SELECT id FROM file_index WHERE filename = ?", (pdf_path.name,))
    row = cursor.fetchone()
    if not row:
        print(f"[!] file_index kaydı bulunamadı: {pdf_path.name}")
        return
    file_id = row[0]

    print(f"[+] {pdf_path.name} dosyası açıldı. Görseller çıkarılıyor...")

    image_counter = 0

    # Çok büyük pixmap riskine karşı limit (piksel sayısı)
    MAX_PIXELS = 40_000_000  # ~40 MP

    for page_no, page in enumerate(doc, start=1):
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]

            # ---- RAW HASH (dedup için) ----
            try:
                raw_bytes = doc.extract_image(xref)["image"]
            except Exception:
                continue
            sha_raw = sha256_bytes(raw_bytes)

            # ---- GÖRÜNEN HAL (OCR/similarity/CNN için) ----
            rects = page.get_image_rects(xref)
            if not rects:
                continue

            for rect_i, rect in enumerate(rects):
                safe_rect = safe_clip_rect(page, rect)
                if safe_rect is None:
                    continue

                # Önce 2x dene, patlarsa 1x'e düş
                render_bytes = None
                used_scale = None

                for scale in (2, 1):
                    try:
                        m = fitz.Matrix(scale, scale)
                        pix = page.get_pixmap(matrix=m, clip=safe_rect, alpha=False)

                        if pix.width <= 1 or pix.height <= 1:
                            continue
                        if pix.width * pix.height > MAX_PIXELS:
                            continue

                        render_bytes = pix.tobytes("png")
                        used_scale = scale
                        break
                    except Exception:
                        render_bytes = None

                if render_bytes is None:
                    # Bu rect problemli; tüm PDF'i patlatma, sadece bunu atla
                    continue

                sha_render = sha256_bytes(render_bytes)

                # image_index: eski mantığın kalsın ama rect_i ile benzersiz yap
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

                # Thumbnail
                try:
                    image = Image.open(io.BytesIO(render_bytes))
                    image.thumbnail((128, 128))
                    thumb_name = f"{pdf_path.stem}_p{page_no}_{img_index}_{rect_i}.png"
                    image.save(OUT_DIR / thumb_name)
                except Exception:
                    # Thumbnail bozulsa bile ana akışı durdurma
                    pass

                image_counter += 1

    conn.commit()
    print(f"[✓] {pdf_path.name} içinden {image_counter} görsel çıkarıldı.")


def process_all_pdfs():
    """data/ klasöründeki tüm PDF'leri işler."""
    conn = create_connection(DB_PATH)
    try:
        for pdf_file in PDF_DIR.glob("*.pdf"):
            extract_images_from_pdf(pdf_file, conn)
    finally:
        conn.close()
    print("Tüm PDF’lerin görselleri çıkarıldı ve veritabanına kaydedildi.")


if __name__ == "__main__":
    process_all_pdfs()
