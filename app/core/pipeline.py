# app/core/pipeline.py
from __future__ import annotations

import sys
from pathlib import Path
import traceback
from typing import Callable, Any


# ------------------------------------------------------------
# Proje kökünü garanti et (mutlak importlar için)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../ProjectNexus-Intelligent-PDF-Analysis
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_step(name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
    print(f"\n--- [STEP] {name} ---")
    try:
        out = fn(*args, **kwargs)
        print(f"--- [OK]   {name} ---")
        return out
    except Exception as e:
        print(f"\n❌ [FAIL] {name}: {e}")
        traceback.print_exc()
        raise


def _ensure_db_schema() -> None:
    from app.core.db import create_connection, create_tables

    conn = create_connection()
    try:
        create_tables(conn)
    finally:
        conn.close()


def _run_pdf_extract() -> None:
    """
    pdf_extract modülünde fonksiyon adı projeden projeye değişebiliyor.
    Bu yüzden iki olası ismi destekliyoruz.
    """
    from app.core import pdf_extract

    if hasattr(pdf_extract, "process_all_pdfs"):
        pdf_extract.process_all_pdfs()
        return

    if hasattr(pdf_extract, "process_all"):
        pdf_extract.process_all()
        return

    raise AttributeError("app.core.pdf_extract içinde process_all_pdfs veya process_all bulunamadı.")


def _run_text_ingestion() -> None:
    """
    Metin çıkarımı/DB yazımı adımı.
    - Eğer app.text.text_processor içinde run_text_ingestion varsa onu çağır.
    - Yoksa fallback: data/*.pdf üzerinde extract_text_lines çalıştırıp get_or_create_file ile DB'ye yazdır.
    """
    from app.text import text_processor

    if hasattr(text_processor, "run_text_ingestion"):
        text_processor.run_text_ingestion()
        return

    # Fallback
    from app.core.paths import DATA_DIR, DB_PATH
    import sqlite3

    extract_text_lines = getattr(text_processor, "extract_text_lines", None)
    get_or_create_file = getattr(text_processor, "get_or_create_file", None)

    if extract_text_lines is None or get_or_create_file is None:
        raise AttributeError(
            "app.text.text_processor içinde run_text_ingestion yok ve "
            "fallback için extract_text_lines/get_or_create_file da bulunamadı."
        )

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] {DATA_DIR} içinde PDF yok, text ingestion atlandı.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    try:
        for pdf_path in pdf_files:
            # file_index kaydı oluştur/çek
            file_id = get_or_create_file(conn, pdf_path)

            # satırları çıkar (beklenen: [(page_no, line_no, text, length), ...])
            lines = extract_text_lines(pdf_path)

            # text_lines’a yaz (tablo şeması db.py ile uyumlu olmalı)
            cur = conn.cursor()
            for page_no, line_no, text, length in lines:
                cur.execute(
                    """
                    INSERT INTO text_lines(file_id, page_no, line_no, text, length)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (file_id, page_no, line_no, text, length),
                )
            conn.commit()
    finally:
        conn.close()


def _run_doc_classification() -> None:
    # app/script/classify_pdfs.py içinde main() var
    from app.script.classify_pdfs import main as classify_main
    classify_main()


def _run_chessboard_detection() -> None:
    # app/script/compute_chessboard_flags.py içinde run_chessboard_detection veya main olabilir
    from app.script import compute_chessboard_flags

    if hasattr(compute_chessboard_flags, "run_chessboard_detection"):
        compute_chessboard_flags.run_chessboard_detection()
        return
    if hasattr(compute_chessboard_flags, "main"):
        compute_chessboard_flags.main()
        return

    raise AttributeError("app.script.compute_chessboard_flags içinde run_chessboard_detection veya main yok.")


def _run_image_features() -> None:
    # app/image/image_features.py içinde process_all veya process olabilir
    from app.image import image_features

    if hasattr(image_features, "process_all"):
        image_features.process_all()
        return
    if hasattr(image_features, "process"):
        image_features.process()
        return

    raise AttributeError("app.image.image_features içinde process_all veya process yok.")


def _run_ocr() -> None:
    # app/image/ocr_run.py içinde process veya main olabilir
    from app.image import ocr_run

    if hasattr(ocr_run, "process"):
        ocr_run.process()
        return
    if hasattr(ocr_run, "main"):
        ocr_run.main()
        return

    raise AttributeError("app.image.ocr_run içinde process veya main yok.")


def _run_text_similarity() -> None:
    # app/text/text_similarity.py içinde main() varsayımı
    from app.text.text_similarity import main as text_sim_main
    text_sim_main()


def _run_image_similarity() -> None:
    # app/image/image_similarity.py içinde process veya main olabilir
    from app.image import image_similarity

    if hasattr(image_similarity, "process"):
        image_similarity.process()
        return
    if hasattr(image_similarity, "main"):
        image_similarity.main()
        return

    raise AttributeError("app.image.image_similarity içinde process veya main yok.")


def full_pipeline_run() -> None:
    print("=====================================================")
    print("                PROJECT NEXUS PIPELINE               ")
    print("=====================================================")

    run_step("DB schema (create_tables)", _ensure_db_schema)
    run_step("PDF extract (pdf_extract)", _run_pdf_extract)
    run_step("Text ingestion (text_processor)", _run_text_ingestion)

    run_step("Image features (image_features)", _run_image_features)
    run_step("Chessboard detection (compute_chessboard_flags)", _run_chessboard_detection)
    run_step("OCR (ocr_run)", _run_ocr)

    run_step("Document classification (classify_pdfs)", _run_doc_classification)

    run_step("Text similarity (text_similarity)", _run_text_similarity)
    run_step("Image similarity (image_similarity)", _run_image_similarity)

    print("\n=====================================================")
    print("            ✅ PIPELINE BAŞARIYLA TAMAMLANDI          ")
    print("=====================================================")


if __name__ == "__main__":
    full_pipeline_run()
