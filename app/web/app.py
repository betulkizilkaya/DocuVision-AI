# project/app/web/app.py (FINAL - TÜM TABLOLAR İÇİN LIMIT 50)

import sqlite3
from flask import Flask, render_template, g
import os
import base64
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', '..', 'db', 'corpus.sqlite')

app = Flask(__name__)


# --- Database Connection Management ---

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# --- Summary Metrics Function ---

def get_summary_metrics():
    """Calculates and fetches the summary metrics from the database."""
    db = get_db()
    metrics = {
        "Total PDF Count": 0,
        "Total Images": 0,
        "Average Text Similarity (Avg Score)": 0.0,
        "High Similarity Pairs (>90%)": 0
    }

    try:
        metrics["Total PDF Count"] = db.execute("SELECT COUNT(id) FROM file_index").fetchone()[0]
        # DİKKAT: "Total Images" artık 2135 yerine 50 görünebilir, çünkü sorgu limitli.
        # Bu yüzden metrik sorgularını get_table_data'dan ayırdık.
        metrics["Total Images"] = db.execute("SELECT COUNT(id) FROM pdf_images").fetchone()[0]

        avg_score_result = db.execute("SELECT AVG(avg_score) FROM text_similarity").fetchone()[0]
        if avg_score_result is not None:
            metrics["Average Text Similarity (Avg Score)"] = f"{avg_score_result:.3f}"
        else:
            metrics["Average Text Similarity (Avg Score)"] = "N/A"

        high_sim_result = db.execute("SELECT COUNT(id) FROM text_similarity WHERE avg_score > 0.90").fetchone()[0]
        metrics["High Similarity Pairs (>90%)"] = high_sim_result

    except sqlite3.OperationalError as e:
        print(f"Database error while fetching metrics: {e}")
        metrics["Average Text Similarity (Avg Score)"] = "DB Error"

    return metrics


# --- Data Fetching Function (GÜNCELLENDİ) ---

def get_table_data(table_name):
    """Fetches a slice of data (LIMIT 50) for all tables."""
    db = get_db()

    try:
        query = ""

        # image_features tablosu için 1769'dan başla
        if table_name == 'image_features':
            # 1769. kayıttan başlamak için 1768 satır atla (OFFSET)
            query = f"SELECT * FROM {table_name} LIMIT 50 OFFSET 1768"
        else:
            # Diğer tüm tablolar için İLK 50 satırı al
            query = f"SELECT * FROM {table_name} LIMIT 50"

        cursor = db.execute(query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    except sqlite3.OperationalError:
        return [f"{table_name} Table Missing"], [["Data expected..."]]
    except Exception as e:
        return ["Error"], [[f"Could not retrieve data: {e}"]]


# --- Helper Function: Renk Paleti Oluşturma ---
def create_color_palette(colors_data_string):
    if not colors_data_string or colors_data_string == "NULL" or colors_data_string == "N/A":
        return "N/A"
    try:
        rgb_tuples = re.findall(r'\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)', str(colors_data_string))
        if not rgb_tuples:
            return "Invalid Color Data"
    except Exception:
        return "Parsing Error"
    palette_html = '<div style="display: flex; gap: 2px; align-items: center;">'
    for (r, g, b) in rgb_tuples:
        color_css = f"rgb({r}, {g}, {b})"
        palette_html += f'<div style="width: 20px; height: 20px; background-color: {color_css}; border: 1px solid #ccc; border-radius: 3px;" title="{color_css}"></div>'
    palette_html += '</div>'
    return palette_html


# --- Base64/Renk Paleti Formatlama Fonksiyonu ---
def format_visual_columns(table_name, columns, rows):
    try:
        blob_index = columns.index('blob')
    except ValueError:
        blob_index = -1
    try:
        thumb_index = columns.index('thumbnail_base64')
    except ValueError:
        thumb_index = -1
    try:
        colors_index = columns.index('top_colors')
    except ValueError:
        colors_index = -1
    if blob_index == -1 and thumb_index == -1 and colors_index == -1:
        return rows
    new_rows = []
    for row in rows:
        row_list = list(row)
        target_index = blob_index if blob_index != -1 else thumb_index
        if target_index != -1:
            blob_data = row_list[target_index]
            image_tag = "No Image"
            img_style = "width: 80px; height: 80px; object-fit: cover; border-radius: 4px; cursor: pointer;"
            base64_data = ""
            if blob_data:
                if isinstance(blob_data, bytes) and len(blob_data) > 100:
                    base64_data = base64.b64encode(blob_data).decode('utf-8')
                elif isinstance(blob_data, str) and len(blob_data) > 100:
                    base64_data = blob_data
            if base64_data:
                image_tag = f'''
                <a href="#" data-bs-toggle="modal" data-bs-target="#imageModal" data-img-src="data:image/png;base64,{base64_data}">
                    <img src="data:image/png;base64,{base64_data}" alt="Thumbnail" style="{img_style}">
                </a>
                '''
            row_list[target_index] = image_tag
        if colors_index != -1 and table_name == 'image_features':
            colors_data = row_list[colors_index]
            row_list[colors_index] = create_color_palette(colors_data)
        new_rows.append(row_list)
    return new_rows


# --- Data Retrieval and Formatting Function ---
def get_all_tables():
    table_list = [
        "file_index", "text_lines", "text_similarity",
        "pdf_images", "image_features", "image_similarity", "binary_similarity"
    ]
    all_data = {}
    for table in table_list:
        columns, rows = get_table_data(table)
        rows = format_visual_columns(table, columns, rows)
        all_data[table] = {'columns': columns, 'rows': rows}
    return all_data


# --- Main Route (Web Page) ---
@app.route('/')
def index():
    all_data = get_all_tables()
    summary_metrics = get_summary_metrics()

    final_summary = {
        "Total PDF Count": summary_metrics["Total PDF Count"],
        "Similarity Ratios (Avg)": summary_metrics["Average Text Similarity (Avg Score)"],
        "Total Images": summary_metrics["Total Images"],
        "High Similarity Pairs (>90%)": summary_metrics["High Similarity Pairs (>90%)"]
    }

    return render_template('index.html', tables=all_data, summary=final_summary)


# --- Sunucu Başlatma ---
if __name__ == '__main__':
    app.run(debug=True)