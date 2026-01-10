import sqlite3
from flask import Flask, render_template, g, request, url_for, send_from_directory
import os
import base64
import sys
from pathlib import Path
from math import ceil

# Proje kökünü sys.path'e ekle (script olarak çalıştırmak için)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.paths import DB_PATH, ROOT_DIR
from app.core.db import init_db

THUMBNAIL_DIR = ROOT_DIR / "temp" / "images"
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

# FEN sadece bu PDF'ler için gösterilecek
FEN_ENABLED_FILE_IDS = {2, 5, 11, 13, 14}

app = Flask(__name__)

# DB şemasını garanti et
init_db()


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        try:
            print(f"[DB] USING (raw): {str(DB_PATH)}")
            print(f"[DB] USING (repr): {repr(str(DB_PATH))}")
        except Exception:
            pass

        db = g._database = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON;")
    return db


@app.teardown_appcontext
def close_connection(exception=None):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def paginate(total_count: int, page: int, per_page: int):
    total_pages = max(1, ceil(total_count / per_page)) if per_page > 0 else 1
    page = max(1, min(page, total_pages))
    offset = (page - 1) * per_page
    return total_pages, offset, page


def get_summary_metrics():
    db = get_db()
    metrics = {
        "Total PDF Count": "N/A",
        "Total Images": "N/A",
        "Average Text Similarity (Avg Score)": "N/A",
        "Average Image Similarity (Avg Score)": "N/A",
        "High Similarity Pairs (>90%)": "N/A",
        "High Text Similarity Pairs (>90%)": "N/A",
        "High Image Similarity Pairs (>90%)": "N/A"
    }

    try:
        metrics["Total PDF Count"] = db.execute("SELECT COUNT(id) FROM file_index").fetchone()[0]
    except:
        pass

    try:
        metrics["Total Images"] = db.execute("SELECT COUNT(id) FROM pdf_images").fetchone()[0]
    except:
        pass

    try:
        avg_text = db.execute("SELECT AVG(avg_score) FROM text_similarity").fetchone()[0]
        metrics["Average Text Similarity (Avg Score)"] = f"{avg_text:.3f}" if avg_text is not None else "N/A"
    except:
        pass

    try:
        avg_img = db.execute("SELECT AVG(ssim) FROM image_similarity").fetchone()[0]
        metrics["Average Image Similarity (Avg Score)"] = f"{avg_img:.3f}" if avg_img is not None else "N/A"
    except:
        try:
            avg_img = db.execute("SELECT AVG(avg_similarity) FROM image_similarity").fetchone()[0]
            metrics["Average Image Similarity (Avg Score)"] = f"{avg_img:.3f}" if avg_img is not None else "N/A"
        except:
            pass

    try:
        high_text = db.execute(
            "SELECT COUNT(id) FROM text_similarity WHERE CAST(avg_score AS REAL) > 0.90"
        ).fetchone()[0]
        metrics["High Text Similarity Pairs (>90%)"] = high_text
        metrics["High Similarity Pairs (>90%)"] = high_text
    except:
        pass

    try:
        high_img = db.execute(
            "SELECT COUNT(id) FROM image_similarity WHERE CAST(ssim AS REAL) > 0.90"
        ).fetchone()[0]
        metrics["High Image Similarity Pairs (>90%)"] = high_img
    except:
        try:
            high_img = db.execute(
                "SELECT COUNT(id) FROM image_similarity WHERE CAST(avg_similarity AS REAL) > 0.90"
            ).fetchone()[0]
            metrics["High Image Similarity Pairs (>90%)"] = high_img
        except:
            pass

    return metrics


def get_table_data(table_name):
    db = get_db()
    try:
        query = f"SELECT * FROM {table_name} ORDER BY id ASC LIMIT 500"
        cursor = db.execute(query)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    except sqlite3.OperationalError:
        return [f"{table_name} Missing"], [["Data expected..."]]
    except Exception as e:
        return ["Error"], [[f"Could not retrieve data: {e}"]]


def format_visual_columns(table_name, columns, rows):
    try:
        blob_index = columns.index('blob')
    except:
        blob_index = -1
    try:
        thumb_index = columns.index('thumbnail_base64')
    except:
        thumb_index = -1
    try:
        colors_index = columns.index('top_colors')
    except:
        colors_index = -1

    if blob_index == -1 and thumb_index == -1 and colors_index == -1:
        return rows

    new_rows = []
    for row in rows:
        row = list(row)

        target_index = blob_index if blob_index != -1 else thumb_index
        if target_index != -1:
            blob = row[target_index]
            img_html = "No Image"
            style = "width:80px;height:80px;object-fit:cover;border-radius:4px;cursor:pointer;"
            base64_data = ""
            if blob:
                if isinstance(blob, bytes) and len(blob) > 100:
                    base64_data = base64.b64encode(blob).decode("utf-8")
                elif isinstance(blob, str) and len(blob) > 100:
                    base64_data = blob
            if base64_data:
                img_html = f"""
                <a data-bs-toggle="modal" data-bs-target="#imageModal" data-img-src="data:image/png;base64,{base64_data}">
                    <img src="data:image/png;base64,{base64_data}" style="{style}">
                </a>
                """
            row[target_index] = img_html

        if colors_index != -1 and table_name == "image_features":
            row[colors_index] = row[colors_index] if row[colors_index] else ""

        new_rows.append(row)

    return new_rows


def get_all_tables():
    table_names = [
        "file_index", "text_lines", "pdf_images",
        "text_similarity",
        "image_features", "image_similarity",
        # "binary_similarity",
        "entities_raw", "persons", "person_mentions",
        "chess_fen",
        "ocr_extracts",
    ]

    data = {}
    for table in table_names:
        columns, rows = get_table_data(table)
        rows = format_visual_columns(table, columns, rows)
        data[table] = {"columns": columns, "rows": rows}
    return data


# -------------------------------------------------
# ✅ EKSİK OLAN ROUTE: static_images
# -------------------------------------------------
@app.route('/static/images/<path:filename>')
def static_images(filename):
    return send_from_directory(str(THUMBNAIL_DIR), filename)


@app.route('/pdf/<int:file_id>')
def pdf_detail(file_id):
    db = get_db()

    pdf_row = db.execute(
        "SELECT filename FROM file_index WHERE id = ?",
        (file_id,)
    ).fetchone()

    if pdf_row is None:
        return "404 - PDF Bulunamadı", 404

    pdf_name = pdf_row["filename"]
    pdf_stem = os.path.splitext(pdf_name)[0]

    filter_type = request.args.get("filter", "all")
    sql_condition = ""

    if filter_type == "chessboard":
        sql_condition = "AND T2.is_chessboard = 1"
    elif filter_type == "non_chessboard":
        sql_condition = "AND (T2.is_chessboard = 0 OR T2.is_chessboard IS NULL)"

    fen_enabled = file_id in FEN_ENABLED_FILE_IDS

    # METRİKLER (PDF geneli)
    chess_count = db.execute("""
        SELECT COUNT(*)
        FROM pdf_images T1
        INNER JOIN image_features T2 ON T1.id = T2.image_id
        WHERE T1.file_id = ?
          AND T2.is_chessboard = 1
    """, (file_id,)).fetchone()[0]

    non_chess_count = db.execute("""
        SELECT COUNT(*)
        FROM pdf_images T1
        INNER JOIN image_features T2 ON T1.id = T2.image_id
        WHERE T1.file_id = ?
          AND (T2.is_chessboard = 0 OR T2.is_chessboard IS NULL)
    """, (file_id,)).fetchone()[0]

    ocr_non_chess_count = db.execute("""
        SELECT COUNT(*)
        FROM pdf_images T1
        INNER JOIN image_features T2 ON T1.id = T2.image_id
        LEFT JOIN ocr_extracts OE ON OE.image_id = T1.id
        WHERE T1.file_id = ?
          AND (T2.is_chessboard = 0 OR T2.is_chessboard IS NULL)
          AND OE.text_raw IS NOT NULL
          AND TRIM(OE.text_raw) != ''
    """, (file_id,)).fetchone()[0]

    query = f"""
        SELECT
            T1.id AS image_id,
            T1.page_no,
            T1.image_index,
            T2.is_chessboard,
            T2.chessboard_score,

            CF.fen_format AS fen,
            CF.created_at AS fen_created_at,

            OE.text_raw AS ocr_text

        FROM pdf_images T1
        INNER JOIN image_features T2 ON T1.id = T2.image_id
        LEFT JOIN chess_fen CF ON CF.image_id = T1.id
        LEFT JOIN ocr_extracts OE ON OE.image_id = T1.id

        WHERE T1.file_id = ? {sql_condition}
        ORDER BY T1.page_no, T1.image_index
    """

    rows = db.execute(query, (file_id,)).fetchall()

    processed_images = []
    for r in rows:
        image_index_db = r["image_index"]
        img_index = image_index_db // 1000
        rect_i = image_index_db % 1000

        thumb_name = f"{pdf_stem}_p{r['page_no']}_{img_index}_{rect_i}.png"

        processed_images.append({
            "image_id": r["image_id"],
            "page": r["page_no"],
            "index": r["image_index"],
            "is_chessboard": r["is_chessboard"] if r["is_chessboard"] is not None else 0,
            "score": f"{r['chessboard_score']:.2f}" if r["chessboard_score"] is not None else "0.00",
            "thumbnail_url": url_for("static_images", filename=thumb_name),
            "fen": r["fen"],
            "fen_created_at": r["fen_created_at"],
            "ocr_text": r["ocr_text"],
        })

    return render_template(
        "pdf_detail.html",
        pdf_name=pdf_name,
        images=processed_images,
        file_id=file_id,
        current_filter=filter_type,
        fen_enabled=fen_enabled,
        chess_count=chess_count,
        non_chess_count=non_chess_count,
        ocr_non_chess_count=ocr_non_chess_count
    )


@app.route('/')
def index():
    db = get_db()

    pdf_rows = db.execute(
        "SELECT id, filename, doc_type FROM file_index ORDER BY filename"
    ).fetchall()

    tables = get_all_tables()
    metrics = get_summary_metrics()

    final_summary = {
        "Total PDF Count": metrics["Total PDF Count"],
        "Similarity Ratios (Avg)": metrics["Average Text Similarity (Avg Score)"],
        "Total Images": metrics["Total Images"],
        "High Similarity Pairs (>90%)": metrics["High Similarity Pairs (>90%)"],
        "Total Image Similarity Ratio": metrics["Average Image Similarity (Avg Score)"],
        "High Image Similarity Pairs (>90%)": metrics["High Image Similarity Pairs (>90%)"]
    }

    charts_data = {
        "pdf_labels": ["PDF1", "PDF2", "PDF3", "PDF4"],
        "text_similarity": [78, 85, 92, 88],
        "image_similarity": [65, 90, 75, 80]
    }

    return render_template(
        "index.html",
        tables=tables,
        summary=final_summary,
        charts_data=charts_data,
        pdfs=pdf_rows,
        fen_enabled_ids=FEN_ENABLED_FILE_IDS
    )


if __name__ == '__main__':
    app.run(debug=True)
