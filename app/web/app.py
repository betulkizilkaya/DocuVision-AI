import sqlite3
from flask import Flask, render_template, g, request, url_for, send_from_directory
import os
import base64
import sys
from pathlib import Path
from math import ceil
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.paths import DB_PATH, ROOT_DIR
from app.core.db import init_db

THUMBNAIL_DIR = ROOT_DIR / "temp" / "images"
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

FEN_ENABLED_FILE_IDS = {2, 5, 11, 13, 14}

app = Flask(__name__)
init_db()


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
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
    except Exception:
        pass

    try:
        metrics["Total Images"] = db.execute("SELECT COUNT(id) FROM pdf_images").fetchone()[0]
    except Exception:
        pass

    try:
        avg_text = db.execute("SELECT AVG(avg_score) FROM text_similarity").fetchone()[0]
        metrics["Average Text Similarity (Avg Score)"] = f"{avg_text:.3f}" if avg_text is not None else "N/A"
    except Exception:
        pass

    try:
        avg_img = db.execute("SELECT AVG(ssim) FROM image_similarity").fetchone()[0]
        metrics["Average Image Similarity (Avg Score)"] = f"{avg_img:.3f}" if avg_img is not None else "N/A"
    except Exception:
        try:
            avg_img = db.execute("SELECT AVG(avg_similarity) FROM image_similarity").fetchone()[0]
            metrics["Average Image Similarity (Avg Score)"] = f"{avg_img:.3f}" if avg_img is not None else "N/A"
        except Exception:
            pass

    try:
        high_text = db.execute(
            "SELECT COUNT(id) FROM text_similarity WHERE CAST(avg_score AS REAL) > 0.90"
        ).fetchone()[0]
        metrics["High Text Similarity Pairs (>90%)"] = high_text
        metrics["High Similarity Pairs (>90%)"] = high_text
    except Exception:
        pass

    try:
        high_img = db.execute(
            "SELECT COUNT(id) FROM image_similarity WHERE CAST(ssim AS REAL) > 0.90"
        ).fetchone()[0]
        metrics["High Image Similarity Pairs (>90%)"] = high_img
    except Exception:
        try:
            high_img = db.execute(
                "SELECT COUNT(id) FROM image_similarity WHERE CAST(avg_similarity AS REAL) > 0.90"
            ).fetchone()[0]
            metrics["High Image Similarity Pairs (>90%)"] = high_img
        except Exception:
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
        blob_index = columns.index("blob")
    except Exception:
        blob_index = -1
    try:
        thumb_index = columns.index("thumbnail_base64")
    except Exception:
        thumb_index = -1
    try:
        colors_index = columns.index("top_colors")
    except Exception:
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
        "file_index",
        "text_lines",
        "pdf_images",
        "text_similarity",
        "image_features",
        "image_similarity",
        "entities_raw",
        "persons",
        "person_mentions",
        "chess_fen",
        "chess_fen_multi",
        "final_boards",
        "ocr_extracts",
    ]

    data = {}
    for table in table_names:
        columns, rows = get_table_data(table)
        rows = format_visual_columns(table, columns, rows)
        data[table] = {"columns": columns, "rows": rows}
    return data


@app.route("/static/images/<path:filename>")
def static_images(filename):
    return send_from_directory(str(THUMBNAIL_DIR), filename)


@app.route("/pdf/<int:file_id>")
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
    fen_enabled = file_id in FEN_ENABLED_FILE_IDS

    total_images = db.execute("""
        SELECT COUNT(*)
        FROM pdf_images
        WHERE file_id = ?
    """, (file_id,)).fetchone()[0] or 0

    person_count = db.execute("""
        SELECT COUNT(*)
        FROM pdf_images pi
        LEFT JOIN image_features f ON pi.id = f.image_id
        WHERE pi.file_id = ?
          AND f.predicted_label = 'person'
    """, (file_id,)).fetchone()[0] or 0

    logo_count = db.execute("""
        SELECT COUNT(*)
        FROM pdf_images pi
        LEFT JOIN image_features f ON pi.id = f.image_id
        WHERE pi.file_id = ?
          AND f.predicted_label = 'logo'
    """, (file_id,)).fetchone()[0] or 0

    game_count = db.execute("""
        SELECT COUNT(*)
        FROM pdf_images pi
        LEFT JOIN image_features f ON pi.id = f.image_id
        WHERE pi.file_id = ?
          AND f.predicted_label = 'game_notation'
    """, (file_id,)).fetchone()[0] or 0

    unknown_count = db.execute("""
        SELECT COUNT(*)
        FROM pdf_images pi
        LEFT JOIN image_features f ON pi.id = f.image_id
        WHERE pi.file_id = ?
          AND (f.predicted_label = 'unknown' OR f.predicted_label IS NULL OR TRIM(f.predicted_label) = '')
    """, (file_id,)).fetchone()[0] or 0

    filter_type = request.args.get("filter", "all")
    sql_condition = ""

    if filter_type == "person":
        sql_condition = "AND f.predicted_label = 'person'"
    elif filter_type == "logo":
        sql_condition = "AND f.predicted_label = 'logo'"
    elif filter_type == "game_notation":
        sql_condition = "AND f.predicted_label = 'game_notation'"
    elif filter_type == "unknown":
        sql_condition = "AND (f.predicted_label = 'unknown' OR f.predicted_label IS NULL OR TRIM(f.predicted_label) = '')"

    query = f"""
        SELECT
            pi.id AS image_id,
            pi.page_no,
            pi.image_index,

            f.is_chessboard,
            f.chessboard_score,
            f.has_person,
            f.person_score,
            f.has_logo,
            f.logo_score,
            f.has_game_notation,
            f.game_notation_score,
            f.predicted_label,
            f.predicted_confidence,

            oe.text_raw AS ocr_text

        FROM pdf_images pi
        LEFT JOIN image_features f ON pi.id = f.image_id
        LEFT JOIN ocr_extracts oe ON oe.image_id = pi.id
        WHERE pi.file_id = ? {sql_condition}
        ORDER BY pi.page_no, pi.image_index
    """

    rows = db.execute(query, (file_id,)).fetchall() or []

    # çoklu board + FEN verileri
    multi_rows = db.execute("""
        SELECT
            fb.image_id,
            fb.board_index,
            fb.source,
            fb.clf_score,
            fb.blob_png,
            cfm.fen_format
        FROM final_boards fb
        LEFT JOIN chess_fen_multi cfm
          ON cfm.image_id = fb.image_id AND cfm.board_index = fb.board_index
        WHERE fb.image_id IN (
            SELECT id FROM pdf_images WHERE file_id = ?
        )
        ORDER BY fb.image_id, fb.board_index
    """, (file_id,)).fetchall()

    boards_by_image = defaultdict(list)
    for br in multi_rows:
        board_b64 = ""
        if br["blob_png"]:
            board_b64 = base64.b64encode(br["blob_png"]).decode("utf-8")

        boards_by_image[int(br["image_id"])].append({
            "board_index": int(br["board_index"]) if br["board_index"] is not None else 0,
            "source": br["source"] or "",
            "clf_score": float(br["clf_score"]) if br["clf_score"] is not None else 0.0,
            "fen": br["fen_format"] or "",
            "board_img_b64": board_b64,
        })

    processed_images = []

    for r in rows:
        image_index_db = r["image_index"] if r["image_index"] is not None else 0
        img_index = image_index_db // 1000
        rect_i = image_index_db % 1000
        thumb_name = f"{pdf_stem}_p{r['page_no']}_{img_index}_{rect_i}.png"

        predicted_label = r["predicted_label"] if r["predicted_label"] else "unknown"
        predicted_confidence = float(r["predicted_confidence"]) if r["predicted_confidence"] is not None else 0.0
        person_score = float(r["person_score"]) if r["person_score"] is not None else 0.0
        logo_score = float(r["logo_score"]) if r["logo_score"] is not None else 0.0
        game_score = float(r["game_notation_score"]) if r["game_notation_score"] is not None else 0.0
        ocr_text = r["ocr_text"] if r["ocr_text"] is not None else ""
        person_box_count = 1 if (r["has_person"] == 1 and person_score > 0) else 0

        if predicted_label == "game_notation":
            explanation = "OCR metninde satranç hamlesine benzeyen yapı bulundu. Oyun yazımı sinyali bu etiketi öne çıkardı."
        elif predicted_label == "person":
            explanation = "Görselde kişi tespit edildi. Person score ve alan baskınlığı bu sonucu destekledi."
        elif predicted_label == "logo":
            explanation = "Kompakt görsel yapı ve logo heuristiği bu etiketi destekledi."
        else:
            if person_score > 0 or logo_score > 0 or game_score > 0:
                explanation = "Bazı sinyaller bulundu ancak baskınlık eşiğini geçmediği için sonuç unknown olarak kaldı."
            else:
                explanation = "Model bu görsel için güçlü bir sınıflandırma sinyali bulamadı."

        image_id = int(r["image_id"])
        boards = boards_by_image.get(image_id, [])

        processed_images.append({
            "id": image_id,
            "image_id": image_id,
            "page": r["page_no"] if r["page_no"] is not None else "-",
            "index": r["image_index"] if r["image_index"] is not None else 0,

            "thumbnail_url": url_for("static_images", filename=thumb_name),

            "is_chessboard": r["is_chessboard"] if r["is_chessboard"] is not None else 0,
            "score": f"{float(r['chessboard_score']):.2f}" if r["chessboard_score"] is not None else "0.00",

            "predicted_label": predicted_label,
            "predicted_confidence": predicted_confidence,
            "person_score": person_score,
            "logo_score": logo_score,
            "game_notation_score": game_score,
            "person_area_ratio": 0.0,
            "logo_area_ratio": 0.0,
            "person_box_count": person_box_count,
            "ocr_text": ocr_text,
            "explanation": explanation,

            "boards": boards,
            "board_count": len(boards),
        })

    return render_template(
        "pdf_detail.html",
        pdf_name=pdf_name,
        images=processed_images or [],
        file_id=file_id,
        current_filter=filter_type,
        fen_enabled=fen_enabled,

        chess_count=game_count,
        non_chess_count=unknown_count,
        ocr_non_chess_count=sum(1 for x in processed_images if x["ocr_text"]),

        total_images=total_images,
        person_count=person_count,
        logo_count=logo_count,
        game_count=game_count,
        unknown_count=unknown_count
    )


@app.route("/")
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


if __name__ == "__main__":
    app.run(debug=True)