import sqlite3
import io
from pathlib import Path
from PIL import Image
from collections import Counter

# Yol yapılandırması
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"

# %90 karelik sınırı
SQUARE_TOLERANCE = 0.9

def create_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def get_images(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, blob FROM pdf_images")
    return cur.fetchall()


def is_grayscale(img):
    """Görsel siyah-beyaz mı kontrol eder."""
    # Görseli her durumda RGB'ye dönüştür
    if img.mode != "RGB":
        img = img.convert("RGB")

    pixels = img.getdata()
    sample = list(pixels)[::max(1, len(pixels)//5000)]  # örnekleme (maks 5000 piksel kontrol)

    for r, g, b in sample:
        if abs(r - g) > 3 or abs(g - b) > 3 or abs(r - b) > 3:
            return False
    return True



def get_top_colors(img, n=5):
    """Görseldeki en sık geçen n rengi bulur (RGB modunda)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    pixels = list(img.getdata())
    count = Counter(pixels)
    most_common = count.most_common(n)
    # (renk, oran) listesi
    return [(tuple(c), round(cnt / len(pixels), 4)) for c, cnt in most_common]


def analyze_image(image_bytes):
    """Bir görselin özelliklerini hesaplar."""
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    aspect_ratio = round(width / height, 3)
    ratio = min(width, height) / max(width, height)
    is_square = 1 if ratio >= SQUARE_TOLERANCE else 0
    gray = 1 if is_grayscale(img) else 0
    top_colors = None if gray else str(get_top_colors(img))
    return width, height, aspect_ratio, is_square, gray, top_colors


def insert_features(conn, image_id, features):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO image_features 
        (image_id, width, height, aspect_ratio, is_square, is_grayscale, top_colors)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (image_id, *features))
    conn.commit()


def process_all():
    conn = create_connection()
    images = get_images(conn)
    print(f"[INFO] {len(images)} görsel bulundu. Özellikler hesaplanıyor...")

    for img_id, blob in images:
        features = analyze_image(blob)
        insert_features(conn, img_id, features)
        print(f"→ Görsel {img_id} işlendi.")

    conn.close()
    print("[✓] Tüm görsellerin özellikleri image_features tablosuna eklendi.")


if __name__ == "__main__":
    process_all()
