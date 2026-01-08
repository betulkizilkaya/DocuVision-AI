import sqlite3
from typing import Optional
from pathlib import Path

from app.core.paths import DB_PATH  # Tek doğru DB yolu burası


def create_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    SQLite bağlantısı oluşturur.
    - DB_PATH pathlib.Path -> sqlite3.connect için str'e çevrilir
    - Klasörü garanti eder (paths.py zaten yapıyor ama double-safety)
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(db_path),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,  # Flask dev server/Windows için daha sorunsuz
    )
    conn.row_factory = sqlite3.Row

    # Tavsiye edilen PRAGMA'lar
    conn.execute("PRAGMA foreign_keys=ON;")
    # Okuma/yazma yoğun işlerde daha iyi olabilir (istersen aç)
    # conn.execute("PRAGMA journal_mode=WAL;")

    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """
    Tüm şemayı oluşturur (idempotent).
    Var olan tabloları/indeksleri bozmadan eksikleri tamamlar.
    """
    cursor = conn.cursor()

    cursor.executescript(
        """
        -- -------------------------
        -- FILE INDEX
        -- -------------------------
        CREATE TABLE IF NOT EXISTS file_index(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            path TEXT,
            sha256 TEXT
        );

        -- aynı dosya iki kez eklenmesin diye
        CREATE UNIQUE INDEX IF NOT EXISTS ux_file_name_hash
            ON file_index(filename, sha256);

        -- -------------------------
        -- TEXT LINES
        -- -------------------------
        CREATE TABLE IF NOT EXISTS text_lines(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            page_no INTEGER,
            line_no INTEGER,
            text TEXT NOT NULL,
            length INTEGER,
            FOREIGN KEY (file_id) REFERENCES file_index(id)
        );

        -- -------------------------
        -- TEXT SIMILARITY
        -- -------------------------
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

        -- -------------------------
        -- PDF IMAGES (GÜNCEL: RENDER BLOB KULLANILIYOR)
        -- -------------------------
        CREATE TABLE IF NOT EXISTS pdf_images(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            page_no INTEGER NOT NULL,
            image_index INTEGER NOT NULL,

            -- PDF içi kaynak bilgisi
            xref INTEGER NOT NULL,
            rect_i INTEGER NOT NULL,

            -- HASH'LER
            sha256 TEXT NOT NULL,        -- RENDER (visible) hash
            sha256_raw TEXT NOT NULL,    -- RAW (extract_image) hash

            -- RENDER GÖRSEL (asıl kullanılacak)
            blob BLOB NOT NULL,

            FOREIGN KEY (file_id) REFERENCES file_index(id),
            UNIQUE (file_id, page_no, xref, rect_i)
        );

        -- -------------------------
        -- IMAGE FEATURES
        -- -------------------------
        CREATE TABLE IF NOT EXISTS image_features(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            width INTEGER,
            height INTEGER,
            aspect_ratio REAL,
            is_square INTEGER,
            is_grayscale INTEGER,
            top_colors TEXT
        );

        -- -------------------------
        -- IMAGE SIMILARITY
        -- -------------------------
        CREATE TABLE IF NOT EXISTS image_similarity(
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Karşılaştırılan görseller
            image_id_a INTEGER NOT NULL,
            image_id_b INTEGER NOT NULL,

            -- Ara metrikler (debug / analiz için)
            ssim REAL,
            phash REAL,
            orb REAL,
            akaze REAL,

            -- Nihai karar
            label TEXT NOT NULL,          -- EXACT_DUPLICATE | NEAR_DUPLICATE | SIMILAR | NOT_SIMILAR | LOW_QUALITY
            decision_phase INTEGER NOT NULL,       -- 1=hash, 2=phash, 3=ssim, 4=feature(orb-akaze) (for statistics)
            reason TEXT,                  -- örn: "sha256", "phash+ssim", "ssim+orb"  (human-readable)

            -- FK (opsiyonel ama doğru)
            FOREIGN KEY (image_id_a) REFERENCES pdf_images(id),
            FOREIGN KEY (image_id_b) REFERENCES pdf_images(id),

            -- Aynı çifti ikinci kez yazma
            UNIQUE (image_id_a, image_id_b)
        );

        -- -------------------------
        -- BINARY SIMILARITY
        -- -------------------------
        CREATE TABLE IF NOT EXISTS binary_similarity
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id_a INTEGER,
            file_id_b INTEGER,
            similarity_ratio REAL,
            longest_match_size INTEGER
        );

        -- -------------------------
        -- OCR EXTRACTS
        -- -------------------------
        CREATE TABLE IF NOT EXISTS ocr_extracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER UNIQUE,
            text_raw TEXT,
            FOREIGN KEY(image_id) REFERENCES pdf_images(id)
        );

        -- -------------------------
        -- ENTITIES
        -- -------------------------
        CREATE TABLE IF NOT EXISTS entities_raw(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            line_id INTEGER NOT NULL,
            ent_text TEXT NOT NULL,
            ent_type TEXT NOT NULL,
            norm_text TEXT,
            FOREIGN KEY (line_id) REFERENCES text_lines(id)
        );

        CREATE TABLE IF NOT EXISTS persons(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norm_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS person_mentions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            line_id INTEGER NOT NULL,
            raw_name TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(id),
            FOREIGN KEY (line_id) REFERENCES text_lines(id)
        );

        -- -------------------------
        -- CHESS FEN
        -- -------------------------
        CREATE TABLE IF NOT EXISTS chess_fen(
            file_id INTEGER PRIMARY KEY,
            fen_format TEXT NOT NULL,
            image_blob BLOB,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (file_id) REFERENCES file_index(id)
        );
        """
    )

    # -------------------------
    # ALTER TABLE / MIGRATIONS
    # -------------------------

    # image_features tablosuna is_chessboard ve chessboard_score ekle (yoksa)
    cursor.execute("PRAGMA table_info(image_features)")
    cols = {row[1] for row in cursor.fetchall()}

    if "is_chessboard" not in cols:
        cursor.execute("ALTER TABLE image_features ADD COLUMN is_chessboard INTEGER")

    if "chessboard_score" not in cols:
        cursor.execute("ALTER TABLE image_features ADD COLUMN chessboard_score REAL")

    # file_index tablosuna doc_type ekle (yoksa)
    cursor.execute("PRAGMA table_info(file_index)")
    cols = {row[1] for row in cursor.fetchall()}
    if "doc_type" not in cols:
        cursor.execute("ALTER TABLE file_index ADD COLUMN doc_type TEXT")

    conn.commit()


def init_db(db_path: Path = DB_PATH) -> None:
    """
    DB bağlantısını açar, şemayı kurar ve kapatır.
    """
    conn = create_connection(db_path)
    try:
        create_tables(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"[DB] OK. Schema ensured at: {DB_PATH.resolve()}")
