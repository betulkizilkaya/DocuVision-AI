import sqlite3
from flask import Flask, render_template, g
import os
import base64


# ------------------------------------------------------------
# Dosya Yolları
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, '..', '..', 'db', 'corpus.sqlite')

app = Flask(__name__)

# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        # BURAYA YENİDEN EKLEYİN:
        print(f"HATA OLUŞAN TAM YOL: {DB_PATH}") 
        db = g._database = sqlite3.connect(DB_PATH)  # Hata burada oluşuyor
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# ============================================================
# SUMMARY METRICS
# ============================================================

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

    # Total PDFs
    try:
        metrics["Total PDF Count"] = db.execute("SELECT COUNT(id) FROM file_index").fetchone()[0]
    except:
        pass

    # Total Images
    try:
        metrics["Total Images"] = db.execute("SELECT COUNT(id) FROM pdf_images").fetchone()[0]
    except:
        pass

    # Avg text sim
    try:
        avg_text = db.execute("SELECT AVG(avg_score) FROM text_similarity").fetchone()[0]
        metrics["Average Text Similarity (Avg Score)"] = f"{avg_text:.3f}" if avg_text else "N/A"
    except:
        pass

    # Avg image sim
    try:
        avg_img = db.execute("SELECT AVG(avg_similarity) FROM image_similarity").fetchone()[0]
        metrics["Average Image Similarity (Avg Score)"] = f"{avg_img:.3f}" if avg_img else "N/A"
    except:
        pass

    # High text sim > 0.90
    try:
        high_text = db.execute(
            "SELECT COUNT(id) FROM text_similarity WHERE CAST(avg_score AS REAL) > 0.90"
        ).fetchone()[0]
        metrics["High Text Similarity Pairs (>90%)"] = high_text
        metrics["High Similarity Pairs (>90%)"] = high_text
    except:
        pass

    # High image sim > 0.90
    try:
        high_img = db.execute(
            "SELECT COUNT(id) FROM image_similarity WHERE CAST(avg_similarity AS REAL) > 0.90"
        ).fetchone()[0]
        metrics["High Image Similarity Pairs (>90%)"] = high_img
    except:
        pass

    return metrics

# ============================================================
# GENERIC TABLE FETCHER
# ============================================================

def get_table_data(table_name):
    db = get_db()
    try:
        query = f"SELECT * FROM {table_name} ORDER BY id ASC LIMIT 100"
        cursor = db.execute(query)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    except sqlite3.OperationalError:
        return [f"{table_name} Missing"], [["Data expected..."]]
    except Exception as e:
        return ["Error"], [[f"Could not retrieve data: {e}"]]

# ============================================================
# IMAGE BASE64 + COLOR FORMATTER
# ============================================================

def format_visual_columns(table_name, columns, rows):
    try: blob_index = columns.index('blob')
    except: blob_index = -1
    try: thumb_index = columns.index('thumbnail_base64')
    except: thumb_index = -1
    try: colors_index = columns.index('top_colors')
    except: colors_index = -1

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

        # Color palette
        if colors_index != -1 and table_name == "image_features":
            row[colors_index] = "Color palette here"

        new_rows.append(row)
    return new_rows

# ============================================================
# ALL TABLES GETTER
# ============================================================

def get_all_tables():
    table_names = [
        "file_index", "text_lines", "text_similarity",
        "pdf_images", "image_features", "image_similarity", "binary_similarity"
    ]

    data = {}
    for table in table_names:
        columns, rows = get_table_data(table)
        rows = format_visual_columns(table, columns, rows)
        data[table] = {"columns": columns, "rows": rows}

    return data

# ============================================================
# MAIN ROUTE
# ============================================================

@app.route('/')
def index():
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

    # -----------------------------
    # Chart.js için örnek veri
    # -----------------------------
    charts_data = {
        "pdf_labels": ["PDF1", "PDF2", "PDF3", "PDF4"],
        "text_similarity": [78, 85, 92, 88],
        "image_similarity": [65, 90, 75, 80]
    }

    return render_template(
        "index.html",
        tables=tables,
        summary=final_summary,
        charts_data=charts_data
    )

# ============================================================
# SERVER START
# ============================================================

if __name__ == '__main__':
    app.run(debug=True)
