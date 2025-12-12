import sqlite3
import os
from pathlib import Path

# Bu dosyanın bulunduğu klasör (örnek: app/)
APP_DIR = Path(__file__).resolve().parent

# Proje kökü = app'in bir üstü
ROOT_DIR = APP_DIR.parent

# Veritabanı yolu (app'in dışında, kökte)
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"

def create_connection(db_file=DB_PATH):
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    conn = sqlite3.connect(db_file)
    return conn


def create_tables(conn):
    cursor = conn.cursor()
    cursor.executescript("""
    -- Betul
    CREATE TABLE IF NOT EXISTS file_index(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        path TEXT,
        sha256 TEXT   --dosya tekrarını onler
    );

    --aynı dosya iki kez eklenmesin diye
    CREATE UNIQUE INDEX IF NOT EXISTS ux_file_name_hash
        ON file_index(filename, sha256);

    CREATE TABLE IF NOT EXISTS text_lines(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        page_no INTEGER,
        line_no INTEGER,
        text TEXT NOT NULL,
        length INTEGER,
        FOREIGN KEY (file_id) REFERENCES file_index(id)
    );

    CREATE TABLE IF NOT EXISTS text_similarity(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        line_id_a INTEGER NOT NULL,
        line_id_b INTEGER NOT NULL,
        lev_ratio REAL,
        jaro REAL,
        dice REAL,
        tfidf_cosine REAL,
        jaccard_tokens REAL,
        avg_score REAL,
        passed_threshold INTEGER
    );

    -- Hira
    CREATE TABLE IF NOT EXISTS pdf_images(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER,          
        page_no INTEGER,
        image_index INTEGER,
        sha256 TEXT,     --image tekrarını onler
        blob BLOB
    );

    CREATE TABLE IF NOT EXISTS image_features(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER,         
        width INTEGER,
        height INTEGER,
        aspect_ratio REAL,        ----oran hesap
        is_square INTEGER,        -- %90 kare kurali
        is_grayscale INTEGER,     -- 1 siyah-beyaz, 0 renkli
        top_colors TEXT           -- ilk 5 renk (JSON formatında) (SQLite tuple veya list tanımaz)
    );

    CREATE TABLE IF NOT EXISTS image_similarity(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id_a INTEGER,
        image_id_b INTEGER,
        ssim REAL,
        phash REAL,
        orb REAL,
        akaze REAL,
        avg_similarity REAL
    );

    CREATE TABLE IF NOT EXISTS binary_similarity
    (
        id
        INTEGER
        PRIMARY
        KEY
        AUTOINCREMENT,
        file_id_a
        INTEGER,
        file_id_b
        INTEGER,
        similarity_ratio
        REAL,   -- binary düzeyde hesaplanan benzerlik oranı
        longest_match_size
        INTEGER -- en uzun ortak binary kısmın boyutu (byte)
    );

    CREATE TABLE IF NOT EXISTS ocr_extracts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER UNIQUE,  
        text_raw TEXT,   --düz okuma sonucu alınan metin
        FOREIGN KEY(image_id) REFERENCES pdf_images(id)
    );
    
    
        -- Metinden çıkarılan ham entity’ler (satır bazlı)
    CREATE TABLE IF NOT EXISTS entities_raw(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        line_id INTEGER NOT NULL,
        ent_text TEXT NOT NULL,
        ent_type TEXT NOT NULL,   -- PERSON / ORG vb.
        norm_text TEXT,           -- normalize edilmiş versiyon
        FOREIGN KEY (line_id) REFERENCES text_lines(id)
    );

    -- Normalize edilmiş kişi isimleri (tekilleştirilmiş)
    CREATE TABLE IF NOT EXISTS persons(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        norm_name TEXT NOT NULL UNIQUE
    );

    -- Her mention’ı hangi kişiye bağladık
    CREATE TABLE IF NOT EXISTS person_mentions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        line_id INTEGER NOT NULL,
        raw_name TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES persons(id),
        FOREIGN KEY (line_id) REFERENCES text_lines(id)
    );

    """)
    # file_index'e doc_type kolonu ekle (yoksa)
    cursor.execute("PRAGMA table_info(file_index)")
    cols = {row[1] for row in cursor.fetchall()}
    if "doc_type" not in cols:
        cursor.execute("ALTER TABLE file_index ADD COLUMN doc_type TEXT")

    conn.commit()
    print("Tables created successfully")

if __name__=="__main__":
    conn = create_connection()
    create_tables(conn)
    conn.close()
