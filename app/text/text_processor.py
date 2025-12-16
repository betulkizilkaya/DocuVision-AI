import pdfplumber #PDF’ten metin çıkarmak
import sqlite3
import logging #Gereksiz uyarıları susturmak için
from hashlib import sha256 #Dosyanın benzersiz “parmak izini” (hash) hesaplıyor
from pathlib import Path #Dosya yollarını platformdan bağımsız yönetiyor.

# Sadece hata mesajlarını göster (pdfminer uyarıları susar)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

#Proje yapısına göre otomatik olarak yolları bulur
APP_DIR  = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH  = ROOT_DIR / "db" / "corpus.sqlite"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)#Eğer db/ klasörü yoksa otomatik oluşturur.

#dosya parmak izi oluşturma
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    #Amaç: aynı isimli ama farklı içerikli dosyaları ayırt edebilmek.
    h = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)# 1 MB parça parça okuma
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def extract_text_lines(pdf_path: Path):
    #PDF'ten metni çıkarıp; her satırı döndürüyoruz. 20–300 karakter filtresi uyguladık.
    rows = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):#Her sayfayı açar
            text = page.extract_text()#extract_text() ile sayfadaki metni alır.
            if not text:
                continue# Eğer sayfa sadece görselse (örneğin tarama), text boş gelir.
            for line_no, line in enumerate(text.splitlines(), start=1):#splitlines() → satır satır böler.
                line = " ".join(line.strip().split())# strip() → baş/son boşlukları temizler.
                if 20 <= len(line) <= 300:# 20 ≤ len(line) ≤ 300 → çok kısa başlıklar ve devasa paragraflar filtrelenir.
                    rows.append((page_no, line_no, line, len(line)))#Uygun satırlar bu şekilde kaydedilir.
    return rows

def ensure_schema(conn: sqlite3.Connection):# tablo yapısı
    cur = conn.cursor()
    # PDF’nin adı, yolu ve hash değeri
    # her PDF’ten çıkan satırlar
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS file_index( 
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      filename TEXT,
      path TEXT,
      sha256 TEXT
    );
        
    CREATE TABLE IF NOT EXISTS text_lines(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_id INTEGER,
      page_no INTEGER,
      line_no INTEGER,
      text   TEXT,
      length INTEGER
    );
    CREATE UNIQUE INDEX IF NOT EXISTS ux_file_name_hash
      ON file_index(filename, sha256);
    CREATE INDEX IF NOT EXISTS idx_text_file ON text_lines(file_id);
    """)
    #UNIQUE INDEX → aynı dosyayı iki kez eklemeyi engeller.
    #INDEX → aramaları hızlandırır.
    conn.commit()

def get_or_create_file(conn: sqlite3.Connection, pdf_path: Path) -> int:# Dosya kaydını yönetir
    cur = conn.cursor()
    file_hash = sha256_file(pdf_path)
    # Eğer aynı isim + aynı hash zaten varsa “IGNORE” eder. Yoksa yeni kayıt ekler.
    cur.execute("""
        INSERT OR IGNORE INTO file_index(filename, path, sha256)
        VALUES (?, ?, ?)
    """, (pdf_path.name, str(pdf_path), file_hash))
    # Sonra o PDF’nin id değerini döner
    cur.execute("SELECT id FROM file_index WHERE filename=? AND sha256=?", (pdf_path.name, file_hash))
    return cur.fetchone()[0]

def replace_text_lines(conn: sqlite3.Connection, file_id: int, lines):
    # Aynı PDF için önce eski satırları siler, sonra yeni satırları ekler/idempotent
    #tekrar çalıştırma güvenli
    cur = conn.cursor()
    cur.execute("DELETE FROM text_lines WHERE file_id=?", (file_id,))
    if lines:
        cur.executemany(
            "INSERT INTO text_lines(file_id, page_no, line_no, text, length) VALUES (?, ?, ?, ?, ?)",
            [(file_id, p, ln, t, L) for (p, ln, t, L) in lines]
        )
    conn.commit()

if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise SystemExit(f"❌ data klasörü bulunamadı: {DATA_DIR}")

    conn = sqlite3.connect(str(DB_PATH))
    ensure_schema(conn)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"❌ {DATA_DIR} içinde .pdf bulunamadı.")

    for pdf in pdf_files:
        lines = extract_text_lines(pdf)
        file_id = get_or_create_file(conn, pdf)
        replace_text_lines(conn, file_id, lines)
        print(f"✅ {pdf.name}: {len(lines)} satır kaydedildi.")

    conn.close()
    print("🎯 Tüm PDF'ler işlendi ve her satır ayrı kayıt olarak eklendi.")
