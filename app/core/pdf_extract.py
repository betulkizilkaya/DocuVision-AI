import fitz          # PyMuPDF -> PDF dosyalarini okumak ve icinden gorsel cikarmak için
import sqlite3
import hashlib       # Gorsellerin SHA256 hash degerini uretmek icin
import io, os
from pathlib import Path
from PIL import Image    # Thumbnail (kucuk onizleme) olusturmak icin
from app.core.paths import DB_PATH, ROOT_DIR, DATA_DIR

PDF_DIR = DATA_DIR  # data/ klasöründeki PDF’ler buradan okunacak

# Thumbnail’lerin kaydedileceği klasör (app'in dışında, kökte)
OUT_DIR = ROOT_DIR / "temp" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    """Verilen verinin SHA-256 hash'ini döndürür."""
    return hashlib.sha256(data).hexdigest()

def create_connection(db_file=DB_PATH):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # db/ klasörü garanti
    conn = sqlite3.connect(str(db_file))
    return conn

def extract_images_from_pdf(pdf_path: Path, conn):
    """PDF içindeki tüm görselleri çıkarır ve pdf_images tablosuna kaydeder."""
    doc = fitz.open(pdf_path)
    cursor = conn.cursor()

    # file_index tablosunda dosyayı kaydet (Betül'ün tablosu)
    with open(pdf_path, "rb") as f:           #DF dosyasını "binary read" (rb) modunda açar,
        file_hash = sha256_bytes(f.read())    #tüm içeriğini okur ve SHA-256 parmak izini (file_hash) hesaplar.

    cursor.execute("INSERT OR IGNORE INTO file_index (filename, sha256) VALUES (?, ?)",
                   (pdf_path.name, file_hash))
    conn.commit()

    # file_id'yi al
    cursor.execute("SELECT id FROM file_index WHERE filename = ?", (pdf_path.name,))
    file_id = cursor.fetchone()[0]

    print(f"[+] {pdf_path.name} dosyası açıldı. Görseller çıkarılıyor...")

    image_counter = 0
    for page_no, page in enumerate(doc, start=1):
        images = page.get_images(full=True)          #sayfadaki tüm görselleri alır
        for img_index, img in enumerate(images):
            xref = img[0]                            #img[0], görselin PDF içindeki dahili referans numarasıdır (xref). 0 yazma nedeni sırayla teker teker işlemesi
            base_image = doc.extract_image(xref)     #fitz'e o referans numarasındaki asıl görsel verisini PDF'in içinden söküp almasını söyler
            img_bytes = base_image["image"]          #görselin ham bytes datasını alır
            sha = sha256_bytes(img_bytes)            #görselin kendisine ait SHA-256 parmak izini hesaplar

            # Görseli BLOB olarak kaydet
            cursor.execute("""
                INSERT INTO pdf_images (file_id, page_no, image_index, sha256, blob)
                VALUES (?, ?, ?, ?, ?)
            """, (file_id, page_no, img_index, sha, sqlite3.Binary(img_bytes)))

            # Thumbnail oluştur ve kaydet
            image = Image.open(io.BytesIO(img_bytes))   #bytes verisi PIL in anlayacağı formata sokulur
            image.thumbnail((128, 128))                 #en-boy
            thumb_name = f"{pdf_path.stem}_p{page_no}_{img_index}.png"
            image.save(OUT_DIR / thumb_name)

            image_counter += 1

    conn.commit()
    print(f"[✓] {pdf_path.name} içinden {image_counter} görsel çıkarıldı.")


def process_all_pdfs():
    """data/ klasöründeki tüm PDF'leri işler."""
    conn = create_connection(DB_PATH)
    for pdf_file in PDF_DIR.glob("*.pdf"):        #data/ klasöründeki sonu .pdf ile biten tüm dosyaları bulur.
        extract_images_from_pdf(pdf_file, conn)
    conn.close()
    print("Tüm PDF’lerin görselleri çıkarıldı ve veritabanına kaydedildi.")


if __name__ == "__main__":
    process_all_pdfs()