# ============================================================
# ProjectNexus - Pipeline
# End-to-End veri işleme hattı
# ------------------------------------------------------------
# Aşamalar:
#   1. Veritabanı oluşturma (db.py)
#   2. PDF metin çıkarımı (text_ops.py)
#   3. Görsel çıkarımı (pdf_extract.py)
#   4. Metin benzerlik analizi (text_similarity.py)
#   5. Görsel benzerlik analizi (image_similarity.py)
#   6. Flask web arayüzünü başlatma (app.py)
# ============================================================

from __future__ import annotations
import argparse
import sqlite3
import subprocess
from pathlib import Path
import sys
import time

# --- Yol Ayarları ---
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
DB_PATH  = ROOT / "db" / "corpus.sqlite"

# --- Modüller ---
from text_ops import (
    extract_text_lines,
    ensure_schema as ensure_text_schema,
    get_or_create_file,
    replace_text_lines
)

# Hira'nın modülleri (görseller için)
try:
    from pdf_extract import extract_images_to_db
    from image_similarity import compare_all_images
except ImportError:
    extract_images_to_db = None
    compare_all_images = None

# ------------------------------------------------------------
def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def step_prepare_db(conn):
    """Tabloları oluşturur."""
    ensure_text_schema(conn)
    print("📘 Veritabanı yapısı kontrol edildi / oluşturuldu.")


def step_extract_texts(conn, data_dir):
    """PDF dosyalarındaki metinleri çıkarır."""
    pdfs = sorted(data_dir.glob("*.pdf"))
    total_lines = 0
    for pdf in pdfs:
        lines = extract_text_lines(pdf)
        file_id = get_or_create_file(conn, pdf)
        replace_text_lines(conn, file_id, lines)
        total_lines += len(lines)
        print(f"✅ {pdf.name}: {len(lines)} satır kaydedildi.")
    conn.commit()
    print(f"📄 Toplam {len(pdfs)} PDF işlendi, {total_lines} satır çıkarıldı.")


def step_extract_images(conn, data_dir):
    """PDF içindeki görselleri çıkarır (Hira'nın modülü)."""
    if extract_images_to_db is None:
        print("⚠️ Görsel çıkarım modülü bulunamadı (pdf_extract.py eksik).")
        return
    extract_images_to_db(data_dir, conn)
    print("🖼️ PDF görselleri çıkarıldı ve veritabanına kaydedildi.")


def step_text_similarity(threshold=0.90):
    """Metin benzerlik analizi çalıştırır (Betül'ün modülü)."""
    import text_similarity as ts
    old = ts.THRESH
    ts.THRESH = threshold
    ts.main()
    ts.THRESH = old
    print("📊 Metin benzerlik analizi tamamlandı.")


def step_image_similarity(conn):
    """Görseller arası benzerlik (Hira'nın modülü)."""
    if compare_all_images is None:
        print("⚠️ Görsel benzerlik modülü bulunamadı.")
        return
    compare_all_images(conn)
    print("🎨 Görsel benzerlik analizi tamamlandı.")


def step_run_web():
    """Selin'in Flask arayüzünü başlatır."""
    web_app = HERE / "web" / "app.py"
    if web_app.exists():
        print("🌐 Flask web arayüzü başlatılıyor...")
        subprocess.run(["python", str(web_app)])
    else:
        print("⚠️ Flask uygulaması bulunamadı (app/web/app.py eksik).")


# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ProjectNexus – PDF Metin ve Görsel Analiz Pipeline")
    parser.add_argument("--no-sim", action="store_true", help="Benzerlik adımlarını atla")
    parser.add_argument("--no-web", action="store_true", help="Web arayüzünü başlatma")
    parser.add_argument("--threshold", type=float, default=0.90, help="Benzerlik eşiği (varsayılan: 0.90)")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"❌ PDF klasörü bulunamadı: {DATA_DIR}")
        sys.exit(1)

    start_time = time.time()
    conn = connect(DB_PATH)

    try:
        step_prepare_db(conn)
        step_extract_texts(conn, DATA_DIR)
        step_extract_images(conn, DATA_DIR)

        if not args.no_sim:
            step_text_similarity(threshold=args.threshold)
            step_image_similarity(conn)
        else:
            print("⏭️ Benzerlik adımları atlandı (--no-sim seçeneği).")

    finally:
        conn.close()

    if not args.no_web:
        step_run_web()

    print(f"✅ Tüm işlem tamamlandı ({time.time() - start_time:.1f} sn)")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
