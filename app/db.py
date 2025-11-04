import sqlite3
import os
from pathlib import Path

# Bu dosyanın bulunduğu klasör (örnek: app/)
# Not: __file__ yerine Path(__file__) kullanmak daha Pythonic'tir.
APP_DIR = Path(os.path.abspath(__file__)).parent

# Proje kökü = app'in bir üstü
ROOT_DIR = APP_DIR.parent

# Veritabanı yolu (app'in dışında, kökte)
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"


def create_connection(db_file=DB_PATH):
    # Dizin yoksa oluştur
    os.makedirs(os.path.dirname(db_file), exist_ok=True)

    # Foreign Keys desteği için isolation_level=None kullanıldı
    conn = sqlite3.connect(db_file, isolation_level=None)

    # Bağlantı açılır açılmaz Foreign Keys kontrolünü aktif et
    conn.execute("PRAGMA foreign_keys = ON;")

    return conn


def create_tables(conn):
    cursor = conn.cursor()
    cursor.executescript("""
    -- PRAGMA foreign_keys = ON; -- Artık create_connection içinde ayarlandı

    -- Betul
    CREATE TABLE IF NOT EXISTS file_index(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        path TEXT,
        sha256 TEXT UNIQUE NOT NULL
    );

    -- aynı dosya iki kez eklenmesin diye
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
        avg_score REAL,
        FOREIGN KEY (line_id_a) REFERENCES text_lines(id),
        FOREIGN KEY (line_id_b) REFERENCES text_lines(id),
        UNIQUE(line_id_a, line_id_b)
    );

    -- Hira
    CREATE TABLE IF NOT EXISTS pdf_images(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,          
        page_no INTEGER,
        image_index INTEGER,
        sha256 TEXT UNIQUE NOT NULL,     -- image tekrarını onler
        blob BLOB,
        FOREIGN KEY (file_id) REFERENCES file_index(id)
    );

    CREATE TABLE IF NOT EXISTS image_features(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,         
        width INTEGER,
        height INTEGER,
        aspect_ratio REAL,        
        is_square INTEGER,        
        is_grayscale INTEGER,     
        top_colors TEXT,           
        FOREIGN KEY (image_id) REFERENCES pdf_images(id)
    );

    CREATE TABLE IF NOT EXISTS image_similarity(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id_a INTEGER NOT NULL,
        image_id_b INTEGER NOT NULL,
        ssim REAL,
        phash REAL,
        orb REAL,
        avg_similarity REAL,
        FOREIGN KEY (image_id_a) REFERENCES pdf_images(id),
        FOREIGN KEY (image_id_b) REFERENCES pdf_images(id),
        UNIQUE(image_id_a, image_id_b)
    );

    CREATE TABLE IF NOT EXISTS binary_similarity(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id_a INTEGER NOT NULL,        
        file_id_b INTEGER NOT NULL,        
        similarity_ratio REAL,     
        FOREIGN KEY (file_id_a) REFERENCES file_index(id),
        FOREIGN KEY (file_id_b) REFERENCES file_index(id),
        UNIQUE(file_id_a, file_id_b)
    );
    """)
    conn.commit()
    print("Tables created successfully")


if __name__ == "__main__":
    try:
        conn = create_connection()
        create_tables(conn)
    finally:
        if 'conn' in locals() and conn:
            conn.close()