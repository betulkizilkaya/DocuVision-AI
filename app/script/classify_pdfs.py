# app/classify_pdfs.py

from pathlib import Path
import sqlite3

from app import predict_doc_type, load_model
from app import (
    DATA_DIR,          # data klasörünün yolu
    DB_PATH,           # db/corpus.sqlite yolu
    extract_text_lines,
    get_or_create_file,
)

def ensure_doc_type_column(conn: sqlite3.Connection):
    """file_index tablosunda doc_type kolonu yoksa ekler."""
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE file_index ADD COLUMN doc_type TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        # Kolon zaten varsa buraya düşer, sorun değil.
        pass

def extract_full_text(pdf_path: Path) -> str:
    """
    text_ops.extract_text_lines çıktısındaki satırları birleştirip
    tek bir büyük metin haline getirir.
    """
    lines = extract_text_lines(pdf_path)  # [(page_no, line_no, text, length), ...]
    return "\n".join(text for (_p, _ln, text, _L) in lines)

def main():
    # Modeli yükle (eğitimde nereye kaydettiysen aynı path'i kullanıyor olmalı)
    model = load_model()

    conn = sqlite3.connect(str(DB_PATH))
    ensure_doc_type_column(conn)
    cur = conn.cursor()

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"❌ {DATA_DIR} içinde .pdf dosyası bulunamadı.")

    for pdf_path in pdf_files:
        print(f"[PDF] {pdf_path.name} sınıflandırılıyor...")

        full_text = extract_full_text(pdf_path)
        if not full_text.strip():
            print("   -> Metin çıkarılamadı (tamamen görsel olabilir), atlanıyor.")
            continue

        # Modelden tahmin al
        label = predict_doc_type(full_text, model=model)
        print(f"   -> Tahmin edilen tür: {label}")

        # file_index kaydını bul / oluştur
        file_id = get_or_create_file(conn, pdf_path)

        # doc_type alanını güncelle
        cur.execute(
            "UPDATE file_index SET doc_type = ? WHERE id = ?",
            (label, file_id),
        )
        conn.commit()

    conn.close()
    print("\n[OK] Tüm PDF'ler sınıflandırıldı ve veritabanına yazıldı.")

if __name__ == "__main__":
    main()
