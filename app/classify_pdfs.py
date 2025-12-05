# app/classify_pdfs.py

import logging
import sqlite3
from pathlib import Path

import pdfplumber

from app.doc_classifier import load_model, predict_doc_type

# Proje kökü, data ve db yolları
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"


def ensure_doc_type_column(conn):
    """
    file_index tablosunda doc_type kolonu yoksa ekler.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(file_index)")
    cols = [row[1] for row in cur.fetchall()]

    if "doc_type" not in cols:
        print("[DB] file_index tablosuna doc_type kolonu ekleniyor...")
        cur.execute("ALTER TABLE file_index ADD COLUMN doc_type TEXT")
        conn.commit()
        print("[DB] doc_type kolonu eklendi.")


def main():
    # pdfminer uyarı spam'ini susturmak istersen:
    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    # Modeli yükle
    model = load_model()

    # Veritabanına bağlan
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Gerekirse doc_type kolonunu ekle
    ensure_doc_type_column(conn)

    # data klasöründeki tüm pdf'leri tara
    for pdf_path in DATA_DIR.glob("*.pdf"):
        print(f"[PDF] {pdf_path.name} sınıflandırılıyor...")

        # PDF'ten metni çıkar
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        # Tür tahmini
        label = predict_doc_type(text, model=model)
        print(f"   -> Tahmin edilen tür: {label}")

        # DB'de ilgili kaydı güncelle
        # Burada file_index.filename kolonu PDF dosya adı ile eşleşiyor varsayıyoruz
        cur.execute(
            "UPDATE file_index SET doc_type = ? WHERE filename = ?",
            (label, pdf_path.name)
        )

    conn.commit()
    conn.close()
    print("\n[OK] Tüm PDF'ler için doc_type DB'ye kaydedildi.")


if __name__ == "__main__":
    main()
