# app/core/pipeline.py (NİHAİ VE KESİNLİKLE ÇALIŞMASI GEREKEN VERSİYON)

import os
import sys

# --- YOL MANİPÜLASYONU (Tüm hataları atlamak için) ---
# BASE_DIR: app/core/ dizinini verir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR: Projenin kök dizinini verir (ProjectNexus-Intelligent-PDF-Analysis/)
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
# Projenin kök dizinini Python'ın arama yoluna (sys.path) ekliyoruz.
# Bu, "app.text...." gibi mutlak import'ların çalışmasını sağlar.
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


# --- TÜM MODÜLLERİ STANDART İÇE AKTARMA ---
try:
    # 1. db.py (app/core/db.py)
    from app.core.db import create_connection, create_tables

    # 2. pdf_extract.py (app/core/pdf_extract.py)
    from app.core.pdf_extract import process_all_pdfs

    # 3. text_processor.py (YOL: app.text - Son yeniden adlandırma)
    # Varsayım: document_classifier.py içindeki fonksiyonları da kullanıyoruz, bu yüzden onu da import ediyoruz.
    # Bu, 'predict_doc_type' hatasının ana kaynağını çözmelidir.
    from app.text.document_classifier import (
        predict_doc_type,
        load_model
    )
    from app.text.text_processor import extract_text_lines as run_text_ingestion

    # 4. image_features.py (YOL: app.image)
    from app.image.image_features import process_all as run_image_features

    # 5. compute_chessboard_flags.py (YOL: app.script)
    from app.script.compute_chessboard_flags import run_chessboard_detection

    # 6. ocr_run.py (YOL: app.image)
    from app.image.ocr_run import process as run_ocr_process

    # 7. classify_pdfs.py (YOL: app.script)
    from app.script.classify_pdfs import main as run_doc_classification

    # 8. text_similarity.py (YOL: app.text)
    from app.text.text_similarity import main as run_text_similarity

    # 9. image_similarity.py (YOL: app.image)
    from app.image.image_similarity import process as run_image_similarity

except Exception as e:
    print(f"\n❌ KRİTİK HATA: Pipeline başlangıcında modül yüklenemedi: {e}")
    print("Lütfen yukarıdaki modüllerin (app.text, app.image vb.) doğru klasörde ve dosya adlarının (text_processor.py) doğru olduğundan emin olun.")
    exit(1)


# --- FULL PIPELINE FONKSİYONU ---

def full_pipeline_run():
    print("=====================================================")
    print("           ✨ PROJE ANALİZ PIPELINE BAŞLADI ✨         ")
    print("=====================================================")

    # 0. Şema Garantisi
    conn = create_connection()
    create_tables(conn)
    conn.close()

    # 1. Ham Veri Alımı (Önceki adımlar)
    process_all_pdfs()
    run_text_ingestion()

    # 2. Temel Özellikler
    run_image_features()

    # 3. Model Çıkarımları
    run_chessboard_detection() # Bu adım ML modeline (joblib) ihtiyaç duyar
    run_ocr_process()          # Bu adım Tesseract'a ve pytesseract'a ihtiyaç duyar
    run_doc_classification()

    # 4. Detaylı Analizler
    run_text_similarity()
    run_image_similarity()

    print("=====================================================")
    print("             🎉 PIPELINE BAŞARIYLA TAMAMLANDI 🎉       ")
    print("=====================================================")


if __name__ == "__main__":
    full_pipeline_run()