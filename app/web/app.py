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


# -----------------------------------------------------------------------------
# Safe DB helpers
# -----------------------------------------------------------------------------

def table_exists(db: sqlite3.Connection, table_name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def column_exists(db: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    if not table_exists(db, table_name):
        return False

    rows = db.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def get_first_column(db: sqlite3.Connection, table_name: str):
    if not table_exists(db, table_name):
        return None

    rows = db.execute(f"PRAGMA table_info({table_name})").fetchall()
    if not rows:
        return None

    return rows[0][1]


def scalar(db: sqlite3.Connection, query: str, params=(), default=None):
    try:
        row = db.execute(query, params).fetchone()
        if row is None:
            return default
        value = row[0]
        return default if value is None else value
    except Exception:
        return default


def to_float(value, default=None):
    if value is None or value == "N/A":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ratio_to_percent(value, digits: int = 1):
    """
    DB similarity değerleri normalde 0-1 aralığında tutuluyor.
    0.822 -> 82.2
    None gelirse None döndürür. Böylece grafiklerde veri yoksa 0 gibi görünmez.
    """
    num = to_float(value, default=None)

    if num is None:
        return None

    if 0 <= num <= 1:
        num *= 100

    return round(max(0.0, min(num, 100.0)), digits)


def ratio_to_percent_safe(value, digits: int = 1) -> float:
    percent = ratio_to_percent(value, digits)
    return 0.0 if percent is None else percent


def format_percent(value, digits: int = 1) -> str:
    percent = ratio_to_percent(value, digits)
    if percent is None:
        return "N/A"
    return f"{percent:.{digits}f}%"


def format_percent_number(value, digits: int = 1) -> str:
    percent = ratio_to_percent(value, digits)
    if percent is None:
        return "0.0"
    return f"{percent:.{digits}f}"


def format_large_number(value) -> str:
    if value is None or value == "N/A":
        return "N/A"

    try:
        value = int(value)
    except (TypeError, ValueError):
        return str(value)

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return str(value)


def get_similarity_column(db: sqlite3.Connection, table_name: str):
    if table_name == "text_similarity":
        for col in ("avg_score", "tfidf_cosine", "jaccard_tokens", "lev_ratio"):
            if column_exists(db, table_name, col):
                return col

    if table_name == "image_similarity":
        for col in ("ssim", "avg_similarity", "phash", "orb", "akaze"):
            if column_exists(db, table_name, col):
                return col

    return None


# -----------------------------------------------------------------------------
# Enterprise dashboard metrics
# -----------------------------------------------------------------------------

def get_text_document_pair_metrics(db: sqlite3.Connection):
    """
    text_similarity tablosu satır çifti bazlıdır.
    Dashboard için daha doğru metrik:
    - Satır çifti sayısı yerine benzersiz PDF çifti sayılır.
    - Her PDF çifti için maksimum similarity skoru alınır.
    - Ortalama risk, bu PDF çifti skorlarının ortalamasıdır.
    """
    result = {
        "avg_document_pair_score": None,
        "unique_doc_pairs_90": 0,
        "unique_doc_pairs_95": 0,
        "unique_doc_pairs_98": 0,
        "raw_line_matches_90": 0,
        "raw_line_matches_95": 0,
        "raw_line_matches_98": 0,
    }

    if not (
        table_exists(db, "text_similarity")
        and table_exists(db, "text_lines")
    ):
        return result

    score_col = get_similarity_column(db, "text_similarity")
    if not score_col:
        return result

    for threshold, key in [
        (0.90, "raw_line_matches_90"),
        (0.95, "raw_line_matches_95"),
        (0.98, "raw_line_matches_98"),
    ]:
        result[key] = scalar(
            db,
            f"""
            SELECT COUNT(*)
            FROM text_similarity
            WHERE CAST({score_col} AS REAL) > ?
            """,
            (threshold,),
            default=0,
        ) or 0

    try:
        pair_rows = db.execute(
            f"""
            WITH doc_pairs AS (
                SELECT
                    CASE
                        WHEN la.file_id < lb.file_id THEN la.file_id
                        ELSE lb.file_id
                    END AS file_a,
                    CASE
                        WHEN la.file_id < lb.file_id THEN lb.file_id
                        ELSE la.file_id
                    END AS file_b,
                    MAX(CAST(ts.{score_col} AS REAL)) AS pair_score
                FROM text_similarity ts
                JOIN text_lines la ON la.id = ts.line_id_a
                JOIN text_lines lb ON lb.id = ts.line_id_b
                WHERE la.file_id != lb.file_id
                GROUP BY file_a, file_b
            )
            SELECT
                AVG(pair_score) AS avg_pair_score,
                SUM(CASE WHEN pair_score > 0.90 THEN 1 ELSE 0 END) AS pairs_90,
                SUM(CASE WHEN pair_score > 0.95 THEN 1 ELSE 0 END) AS pairs_95,
                SUM(CASE WHEN pair_score > 0.98 THEN 1 ELSE 0 END) AS pairs_98
            FROM doc_pairs
            """
        ).fetchone()

        if pair_rows:
            result["avg_document_pair_score"] = pair_rows["avg_pair_score"]
            result["unique_doc_pairs_90"] = pair_rows["pairs_90"] or 0
            result["unique_doc_pairs_95"] = pair_rows["pairs_95"] or 0
            result["unique_doc_pairs_98"] = pair_rows["pairs_98"] or 0

    except Exception:
        pass

    return result


def get_image_pair_metrics(db: sqlite3.Connection):
    """
    image_similarity görsel çifti bazlıdır.
    Ek olarak benzersiz PDF çifti riski de hesaplanır.
    """
    result = {
        "avg_image_score": None,
        "image_pairs_90": 0,
        "image_pairs_95": 0,
        "image_pairs_98": 0,
        "unique_doc_pairs_90": 0,
        "unique_doc_pairs_95": 0,
        "unique_doc_pairs_98": 0,
    }

    if not table_exists(db, "image_similarity"):
        return result

    score_col = get_similarity_column(db, "image_similarity")
    if not score_col:
        return result

    result["avg_image_score"] = scalar(
        db,
        f"SELECT AVG(CAST({score_col} AS REAL)) FROM image_similarity",
        default=None,
    )

    for threshold, key in [
        (0.90, "image_pairs_90"),
        (0.95, "image_pairs_95"),
        (0.98, "image_pairs_98"),
    ]:
        result[key] = scalar(
            db,
            f"""
            SELECT COUNT(*)
            FROM image_similarity
            WHERE CAST({score_col} AS REAL) > ?
            """,
            (threshold,),
            default=0,
        ) or 0

    if table_exists(db, "pdf_images"):
        try:
            pair_rows = db.execute(
                f"""
                WITH doc_pairs AS (
                    SELECT
                        CASE
                            WHEN ia.file_id < ib.file_id THEN ia.file_id
                            ELSE ib.file_id
                        END AS file_a,
                        CASE
                            WHEN ia.file_id < ib.file_id THEN ib.file_id
                            ELSE ia.file_id
                        END AS file_b,
                        MAX(CAST(sim.{score_col} AS REAL)) AS pair_score
                    FROM image_similarity sim
                    JOIN pdf_images ia ON ia.id = sim.image_id_a
                    JOIN pdf_images ib ON ib.id = sim.image_id_b
                    WHERE ia.file_id != ib.file_id
                    GROUP BY file_a, file_b
                )
                SELECT
                    SUM(CASE WHEN pair_score > 0.90 THEN 1 ELSE 0 END) AS pairs_90,
                    SUM(CASE WHEN pair_score > 0.95 THEN 1 ELSE 0 END) AS pairs_95,
                    SUM(CASE WHEN pair_score > 0.98 THEN 1 ELSE 0 END) AS pairs_98
                FROM doc_pairs
                """
            ).fetchone()

            if pair_rows:
                result["unique_doc_pairs_90"] = pair_rows["pairs_90"] or 0
                result["unique_doc_pairs_95"] = pair_rows["pairs_95"] or 0
                result["unique_doc_pairs_98"] = pair_rows["pairs_98"] or 0

        except Exception:
            pass

    return result


def get_summary_metrics():
    db = get_db()

    text_metrics = get_text_document_pair_metrics(db)
    image_metrics = get_image_pair_metrics(db)

    metrics = {
        "Total PDF Count": scalar(db, "SELECT COUNT(*) FROM file_index", default=0) or 0,
        "Total Images": scalar(db, "SELECT COUNT(*) FROM pdf_images", default=0) or 0,

        "Average Text Similarity (Avg Score)": text_metrics["avg_document_pair_score"],
        "High Similarity Pairs (>90%)": text_metrics["unique_doc_pairs_90"],
        "High Text Similarity Pairs (>90%)": text_metrics["unique_doc_pairs_90"],
        "High Text Similarity Pairs (>95%)": text_metrics["unique_doc_pairs_95"],
        "High Text Similarity Pairs (>98%)": text_metrics["unique_doc_pairs_98"],

        "Raw Text Line Matches (>90%)": text_metrics["raw_line_matches_90"],
        "Raw Text Line Matches (>95%)": text_metrics["raw_line_matches_95"],
        "Raw Text Line Matches (>98%)": text_metrics["raw_line_matches_98"],

        "Average Image Similarity (Avg Score)": image_metrics["avg_image_score"],
        "High Image Similarity Pairs (>90%)": image_metrics["image_pairs_90"],
        "High Image Similarity Pairs (>95%)": image_metrics["image_pairs_95"],
        "High Image Similarity Pairs (>98%)": image_metrics["image_pairs_98"],

        "High Image Document Pairs (>90%)": image_metrics["unique_doc_pairs_90"],
        "High Image Document Pairs (>95%)": image_metrics["unique_doc_pairs_95"],
        "High Image Document Pairs (>98%)": image_metrics["unique_doc_pairs_98"],
    }

    return metrics


def get_per_document_text_score(db: sqlite3.Connection, file_id: int, score_col: str):
    if not (table_exists(db, "text_similarity") and table_exists(db, "text_lines")):
        return None

    return scalar(
        db,
        f"""
        SELECT MAX(CAST(ts.{score_col} AS REAL))
        FROM text_similarity ts
        JOIN text_lines la ON la.id = ts.line_id_a
        JOIN text_lines lb ON lb.id = ts.line_id_b
        WHERE la.file_id != lb.file_id
          AND (la.file_id = ? OR lb.file_id = ?)
        """,
        (file_id, file_id),
        default=None,
    )


def get_per_document_image_score(db: sqlite3.Connection, file_id: int, score_col: str):
    if not (table_exists(db, "image_similarity") and table_exists(db, "pdf_images")):
        return None

    return scalar(
        db,
        f"""
        SELECT MAX(CAST(sim.{score_col} AS REAL))
        FROM image_similarity sim
        JOIN pdf_images ia ON ia.id = sim.image_id_a
        JOIN pdf_images ib ON ib.id = sim.image_id_b
        WHERE ia.file_id != ib.file_id
          AND (ia.file_id = ? OR ib.file_id = ?)
        """,
        (file_id, file_id),
        default=None,
    )


def get_real_charts_data(limit: int = 10):
    """
    Chart.js için gerçek veri üretir.

    Önemli düzeltme:
    - Veri yoksa 0 göndermez, None gönderir.
    - Böylece scatter chart'ta image similarity olmayan PDF'ler 0'a yapışmış gibi görünmez.
    - En riskli 10 doküman gösterilir.
    """
    db = get_db()

    pdf_rows = db.execute(
        "SELECT id, filename FROM file_index ORDER BY filename"
    ).fetchall()

    text_col = get_similarity_column(db, "text_similarity")
    img_col = get_similarity_column(db, "image_similarity")

    records = []

    for pdf in pdf_rows:
        file_id = int(pdf["id"])
        filename = pdf["filename"] or f"PDF {file_id}"

        text_score = None
        image_score = None

        if text_col:
            text_score = get_per_document_text_score(db, file_id, text_col)

        if img_col:
            image_score = get_per_document_image_score(db, file_id, img_col)

        text_percent = ratio_to_percent(text_score, 1)
        image_percent = ratio_to_percent(image_score, 1)

        risk_sort_value = max(
            text_percent if text_percent is not None else -1,
            image_percent if image_percent is not None else -1,
        )

        records.append({
            "label": filename if len(filename) <= 28 else filename[:25] + "...",
            "text": text_percent,
            "image": image_percent,
            "risk_sort": risk_sort_value,
        })

    records.sort(key=lambda x: x["risk_sort"], reverse=True)
    records = records[:limit]

    return {
        "pdf_labels": [r["label"] for r in records],
        "text_similarity": [r["text"] for r in records],
        "image_similarity": [r["image"] for r in records],
        "has_text_similarity_data": any(r["text"] is not None for r in records),
        "has_image_similarity_data": any(r["image"] is not None for r in records),
        "chart_metric_note": (
            "Charts show maximum cross-document similarity risk per PDF. "
            "Missing values mean no comparable data was found, not zero similarity."
        ),
    }


def get_fen_enabled_file_ids():
    db = get_db()
    ids = set(FEN_ENABLED_FILE_IDS)

    if table_exists(db, "chess_fen") and table_exists(db, "pdf_images"):
        try:
            rows = db.execute(
                """
                SELECT DISTINCT pi.file_id
                FROM chess_fen cf
                JOIN pdf_images pi ON pi.id = cf.image_id
                """
            ).fetchall()
            ids.update(int(r[0]) for r in rows if r[0] is not None)
        except Exception:
            pass

    if table_exists(db, "chess_fen_multi") and table_exists(db, "pdf_images"):
        try:
            rows = db.execute(
                """
                SELECT DISTINCT pi.file_id
                FROM chess_fen_multi cfm
                JOIN pdf_images pi ON pi.id = cfm.image_id
                """
            ).fetchall()
            ids.update(int(r[0]) for r in rows if r[0] is not None)
        except Exception:
            pass

    return ids


# -----------------------------------------------------------------------------
# Table data
# -----------------------------------------------------------------------------

def get_table_data(table_name):
    db = get_db()

    allowed_tables = {
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
    }

    if table_name not in allowed_tables:
        return ["Error"], [["Invalid table name"]]

    try:
        if not table_exists(db, table_name):
            return [f"{table_name} Missing"], [["Data expected..."]]

        if column_exists(db, table_name, "id"):
            order_col = "id"
        else:
            order_col = get_first_column(db, table_name)

        if order_col:
            query = f"SELECT * FROM {table_name} ORDER BY {order_col} ASC LIMIT 500"
        else:
            query = f"SELECT * FROM {table_name} LIMIT 500"

        cursor = db.execute(query)
        real_columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()

        if table_name == "file_index" and "id" in real_columns:
            id_index = real_columns.index("id")

            display_columns = real_columns.copy()
            display_columns[id_index] = "document_no"

            display_rows = []
            for display_no, row in enumerate(rows, start=1):
                row_list = list(row)
                row_list[id_index] = display_no
                display_rows.append(row_list)

            return display_columns, display_rows

        return real_columns, rows

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
        data[table] = {
            "columns": columns,
            "rows": rows,
        }

    return data


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/static/images/<path:filename>")
def static_images(filename):
    return send_from_directory(str(THUMBNAIL_DIR), filename)


@app.route("/pdf/<int:file_id>")
def pdf_detail(file_id):
    db = get_db()

    pdf_row = db.execute(
        "SELECT filename FROM file_index WHERE id = ?",
        (file_id,),
    ).fetchone()

    if pdf_row is None:
        return "404 - PDF Bulunamadı", 404

    pdf_name = pdf_row["filename"]
    pdf_stem = os.path.splitext(pdf_name)[0]
    fen_enabled = file_id in get_fen_enabled_file_ids()

    total_images = scalar(
        db,
        """
        SELECT COUNT(*)
        FROM pdf_images
        WHERE file_id = ?
        """,
        (file_id,),
        default=0,
    ) or 0

    person_count = scalar(
        db,
        """
        SELECT COUNT(*)
        FROM pdf_images pi
        LEFT JOIN image_features f ON pi.id = f.image_id
        WHERE pi.file_id = ?
          AND f.predicted_label = 'person'
        """,
        (file_id,),
        default=0,
    ) or 0

    logo_count = scalar(
        db,
        """
        SELECT COUNT(*)
        FROM pdf_images pi
        LEFT JOIN image_features f ON pi.id = f.image_id
        WHERE pi.file_id = ?
          AND f.predicted_label = 'logo'
        """,
        (file_id,),
        default=0,
    ) or 0

    game_count = scalar(
        db,
        """
        SELECT COUNT(*)
        FROM pdf_images pi
        LEFT JOIN image_features f ON pi.id = f.image_id
        WHERE pi.file_id = ?
          AND f.predicted_label = 'game_notation'
        """,
        (file_id,),
        default=0,
    ) or 0

    unknown_count = scalar(
        db,
        """
        SELECT COUNT(*)
        FROM pdf_images pi
        LEFT JOIN image_features f ON pi.id = f.image_id
        WHERE pi.file_id = ?
          AND (
            f.predicted_label = 'unknown'
            OR f.predicted_label IS NULL
            OR TRIM(f.predicted_label) = ''
          )
        """,
        (file_id,),
        default=0,
    ) or 0

    filter_type = request.args.get("filter", "all")
    sql_condition = ""

    if filter_type == "person":
        sql_condition = "AND f.predicted_label = 'person'"
    elif filter_type == "logo":
        sql_condition = "AND f.predicted_label = 'logo'"
    elif filter_type == "game_notation":
        sql_condition = "AND f.predicted_label = 'game_notation'"
    elif filter_type == "unknown":
        sql_condition = """
        AND (
            f.predicted_label = 'unknown'
            OR f.predicted_label IS NULL
            OR TRIM(f.predicted_label) = ''
        )
        """

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

    multi_rows = []
    if table_exists(db, "final_boards") and table_exists(db, "chess_fen_multi"):
        try:
            multi_rows = db.execute(
                """
                SELECT
                    fb.image_id,
                    fb.board_index,
                    fb.source,
                    fb.clf_score,
                    fb.blob_png,
                    cfm.fen_format
                FROM final_boards fb
                LEFT JOIN chess_fen_multi cfm
                  ON cfm.image_id = fb.image_id
                 AND cfm.board_index = fb.board_index
                WHERE fb.image_id IN (
                    SELECT id FROM pdf_images WHERE file_id = ?
                )
                ORDER BY fb.image_id, fb.board_index
                """,
                (file_id,),
            ).fetchall()
        except Exception:
            multi_rows = []

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
        unknown_count=unknown_count,
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
        "Total Images": metrics["Total Images"],

        "Similarity Ratios (Avg)": format_percent_number(metrics["Average Text Similarity (Avg Score)"]),
        "Similarity Ratios Display": format_percent(metrics["Average Text Similarity (Avg Score)"]),
        "Similarity Ratios Progress": ratio_to_percent_safe(metrics["Average Text Similarity (Avg Score)"]),
        "Similarity Ratios Raw": metrics["Average Text Similarity (Avg Score)"],

        "Total Image Similarity Ratio": format_percent_number(metrics["Average Image Similarity (Avg Score)"]),
        "Total Image Similarity Display": format_percent(metrics["Average Image Similarity (Avg Score)"]),
        "Total Image Similarity Progress": ratio_to_percent_safe(metrics["Average Image Similarity (Avg Score)"]),
        "Total Image Similarity Raw": metrics["Average Image Similarity (Avg Score)"],

        "High Similarity Pairs (>90%)": format_large_number(metrics["High Similarity Pairs (>90%)"]),
        "High Similarity Pairs (>95%)": format_large_number(metrics["High Text Similarity Pairs (>95%)"]),
        "High Similarity Pairs (>98%)": format_large_number(metrics["High Text Similarity Pairs (>98%)"]),
        "High Similarity Pairs Raw": metrics["High Similarity Pairs (>90%)"],

        "Raw Text Line Matches (>90%)": format_large_number(metrics["Raw Text Line Matches (>90%)"]),
        "Raw Text Line Matches (>95%)": format_large_number(metrics["Raw Text Line Matches (>95%)"]),
        "Raw Text Line Matches (>98%)": format_large_number(metrics["Raw Text Line Matches (>98%)"]),

        "High Image Similarity Pairs (>90%)": format_large_number(metrics["High Image Similarity Pairs (>90%)"]),
        "High Image Similarity Pairs (>95%)": format_large_number(metrics["High Image Similarity Pairs (>95%)"]),
        "High Image Similarity Pairs (>98%)": format_large_number(metrics["High Image Similarity Pairs (>98%)"]),
        "High Image Similarity Pairs Raw": metrics["High Image Similarity Pairs (>90%)"],

        "High Image Document Pairs (>90%)": format_large_number(metrics["High Image Document Pairs (>90%)"]),
        "High Image Document Pairs (>95%)": format_large_number(metrics["High Image Document Pairs (>95%)"]),
        "High Image Document Pairs (>98%)": format_large_number(metrics["High Image Document Pairs (>98%)"]),
    }

    charts_data = get_real_charts_data(limit=10)
    fen_enabled_ids = get_fen_enabled_file_ids()

    return render_template(
        "index.html",
        tables=tables,
        summary=final_summary,
        charts_data=charts_data,
        pdfs=pdf_rows,
        fen_enabled_ids=fen_enabled_ids,
    )


if __name__ == "__main__":
    app.run(debug=True)