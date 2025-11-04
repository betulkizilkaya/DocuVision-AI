# ============================================================
# project/app/web/app.py (FINAL - TÜM TABLOLAR İÇİN LIMIT 50)
# ============================================================
# Bu dosya, Flask web uygulamasının ana sunucu (backend) dosyasıdır.
# Görevi:
#   - SQLite veritabanına bağlanmak,
#   - Farklı tabloların verilerini çekmek (limit 50 satır),
#   - Görsel (image, renk paleti) formatlamasını yapmak,
#   - HTML şablonuna (index.html) bu verileri yollayıp sayfayı üretmektir.
# ============================================================


import sqlite3                 # SQLite veritabanına bağlanmak için.
from flask import Flask, render_template, g  # Flask framework ve context yönetimi.
import os                      # Dosya/dizin yollarını yönetmek için.
import base64                  # Görselleri Base64'e çevirmek için.
import re                      # Renk paletindeki RGB verilerini ayrıştırmak için.

# ------------------------------------------------------------
# Dosya Yollarını Ayarlama
# ------------------------------------------------------------
# BASE_DIR: Bu Python dosyasının bulunduğu dizini temsil eder.
# DB_PATH: Gerçek veritabanı dosyasının konumunu belirtir.
# Burada '../..' ile 2 klasör yukarı çıkarak 'db/corpus.sqlite' dosyasına ulaşıyoruz.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', '..', 'db', 'corpus.sqlite')

# Flask uygulamasını başlatıyoruz.
app = Flask(__name__)


# ============================================================
# --- DATABASE CONNECTION MANAGEMENT ---
# ============================================================
# Flask'ta her istek (request) için ayrı bir veritabanı bağlantısı kullanılır.
# Bağlantıyı "g" adlı global bir Flask context nesnesine koyarız.
# İş bitince bağlantı otomatik kapanır.
# ============================================================

def get_db():
    """Veritabanı bağlantısını oluşturur veya mevcut olanı döndürür."""
    db = getattr(g, '_database', None)  # Daha önce bir bağlantı var mı?
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)  # Yoksa yeni bağlantı oluştur.
        db.row_factory = sqlite3.Row  # Sorgu sonuçlarını dictionary benzeri erişim için ayarlar.
    return db


@app.teardown_appcontext
def close_connection(exception):
    """İstek tamamlanınca (veya hata olunca) veritabanı bağlantısını kapatır."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# ============================================================
# --- SUMMARY METRICS FUNCTION ---
# ============================================================
# Bu fonksiyon, dashboard üst kısmında görülen küçük özet kutularını (metrics) hesaplar:
#   - Toplam PDF sayısı
#   - Toplam görsel (image) sayısı
#   - Ortalama benzerlik skoru
#   - %90 üzeri benzerlik sayısı
# ============================================================

def get_summary_metrics():
    """Veritabanından özet metrikleri hesaplar."""
    db = get_db()  # Veritabanı bağlantısını al.
    metrics = {
        "Total PDF Count": 0,
        "Total Images": 0,
        "Average Text Similarity (Avg Score)": 0.0,
        "High Similarity Pairs (>90%)": 0
    }

    try:
        # --- 1️⃣ Toplam PDF sayısı ---
        metrics["Total PDF Count"] = db.execute("SELECT COUNT(id) FROM file_index").fetchone()[0]

        # --- 2️⃣ Toplam Görsel sayısı ---
        metrics["Total Images"] = db.execute("SELECT COUNT(id) FROM pdf_images").fetchone()[0]

        # --- 3️⃣ Ortalama benzerlik skoru ---
        avg_score_result = db.execute("SELECT AVG(avg_score) FROM text_similarity").fetchone()[0]
        if avg_score_result is not None:
            metrics["Average Text Similarity (Avg Score)"] = f"{avg_score_result:.3f}"
        else:
            metrics["Average Text Similarity (Avg Score)"] = "N/A"

        # --- 4️⃣ Yüksek benzerlik oranı (>90%) ---
        high_sim_result = db.execute("SELECT COUNT(id) FROM text_similarity WHERE avg_score > 0.90").fetchone()[0]
        metrics["High Similarity Pairs (>90%)"] = high_sim_result

    except sqlite3.OperationalError as e:
        # Eğer tablo yoksa veya kolon adı yanlışsa hata oluşur.
        print(f"Database error while fetching metrics: {e}")
        metrics["Average Text Similarity (Avg Score)"] = "DB Error"

    return metrics


# ============================================================
# --- DATA FETCHING FUNCTION ---
# ============================================================
# Her tablo için yalnızca 50 satır veri alınır (LIMIT 50)
# "image_features" tablosu için 1768 kayıt atlanarak (OFFSET 1768) 1769. satırdan başlanır.
# ============================================================

def get_table_data(table_name):
    """Tüm tablolar için ilk 50 satırı (veya belirli offsetten) getirir."""
    db = get_db()
    try:
        query = ""

        # Özel durum: image_features tablosunda ilk 1768 satırı atlıyoruz.
        if table_name == 'image_features':
            query = f"SELECT * FROM {table_name} LIMIT 50 OFFSET 1768"
        else:
            query = f"SELECT * FROM {table_name} LIMIT 50"

        cursor = db.execute(query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return columns, rows

    except sqlite3.OperationalError:
        # Eğer tablo yoksa (örneğin test sırasında)
        return [f"{table_name} Table Missing"], [["Data expected..."]]
    except Exception as e:
        # Diğer beklenmedik hatalar
        return ["Error"], [[f"Could not retrieve data: {e}"]]


# ============================================================
# --- COLOR PALETTE HELPER FUNCTION ---
# ============================================================
# Bu fonksiyon, veritabanındaki "top_colors" sütunundan gelen renk verilerini
# (örneğin "(23,45,67),(200,210,220)") HTML içinde küçük renk kutucuklarına dönüştürür.
# ============================================================

def create_color_palette(colors_data_string):
    """RGB renk dizilerini küçük karelere çevirir."""
    if not colors_data_string or colors_data_string == "NULL" or colors_data_string == "N/A":
        return "N/A"
    try:
        # regex ile "(r,g,b)" kalıplarını yakalar.
        rgb_tuples = re.findall(r'\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)', str(colors_data_string))
        if not rgb_tuples:
            return "Invalid Color Data"
    except Exception:
        return "Parsing Error"

    # HTML içeriğini oluştur
    palette_html = '<div style="display: flex; gap: 2px; align-items: center;">'
    for (r, g, b) in rgb_tuples:
        color_css = f"rgb({r}, {g}, {b})"
        palette_html += f'<div style="width: 20px; height: 20px; background-color: {color_css}; border: 1px solid #ccc; border-radius: 3px;" title="{color_css}"></div>'
    palette_html += '</div>'
    return palette_html


# ============================================================
# --- BASE64 / COLOR PALETTE FORMATLAMA ---
# ============================================================
# Bu fonksiyon, satırlardaki "blob" veya "thumbnail_base64" kolonlarını
# doğrudan img etiketi hâline getirir (HTML).
# Renk paleti de burada görselleştirilir.
# ============================================================

def format_visual_columns(table_name, columns, rows):
    """Görsel verileri (blob/base64) img tag olarak, renkleri palet olarak dönüştürür."""
    # Kolonların index'lerini bul
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

    # Eğer görsel veya renk yoksa direkt döndür.
    if blob_index == -1 and thumb_index == -1 and colors_index == -1:
        return rows

    new_rows = []

    # Her satırı dolaş
    for row in rows:
        row_list = list(row)
        target_index = blob_index if blob_index != -1 else thumb_index

        # Görsel sütunu varsa:
        if target_index != -1:
            blob_data = row_list[target_index]
            image_tag = "No Image"
            img_style = "width: 80px; height: 80px; object-fit: cover; border-radius: 4px; cursor: pointer;"
            base64_data = ""

            # Görsel verisi varsa, base64'e çevir
            if blob_data:
                if isinstance(blob_data, bytes) and len(blob_data) > 100:
                    base64_data = base64.b64encode(blob_data).decode('utf-8')
                elif isinstance(blob_data, str) and len(blob_data) > 100:
                    base64_data = blob_data

            # Eğer base64 verisi oluştuysa img etiketi hazırla
            if base64_data:
                image_tag = f'''
                <a href="#" data-bs-toggle="modal" data-bs-target="#imageModal" data-img-src="data:image/png;base64,{base64_data}">
                    <img src="data:image/png;base64,{base64_data}" alt="Thumbnail" style="{img_style}">
                </a>
                '''
            row_list[target_index] = image_tag

        # Renk paleti sütunu varsa:
        if colors_index != -1 and table_name == 'image_features':
            colors_data = row_list[colors_index]
            row_list[colors_index] = create_color_palette(colors_data)

        new_rows.append(row_list)
    return new_rows


# ============================================================
# --- Tüm Tabloları Al ve Formatla ---
# ============================================================
# Burada hangi tabloların gösterileceğini belirtiyoruz.
# Her tabloyu get_table_data() ile çekip, görselleri formatlayıp
# "all_data" sözlüğüne ekliyoruz.
# ============================================================

def get_all_tables():
    """Tüm tabloların kolon ve satırlarını getirip formatlar."""
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


# ============================================================
# --- MAIN ROUTE (Ana Sayfa) ---
# ============================================================
# Flask’ın ana sayfa rotası.
# Burada tüm tablolar ve özet metrikler çekilip
# 'index.html' adlı template’e yollanır.
# ============================================================

@app.route('/')
def index():
    all_data = get_all_tables()          # Tüm tabloların verilerini çek.
    summary_metrics = get_summary_metrics()  # Özet metrikleri çek.

    # Dashboard kutucukları için sadeleştirilmiş dictionary
    final_summary = {
        "Total PDF Count": summary_metrics["Total PDF Count"],
        "Similarity Ratios (Avg)": summary_metrics["Average Text Similarity (Avg Score)"],
        "Total Images": summary_metrics["Total Images"],
        "High Similarity Pairs (>90%)": summary_metrics["High Similarity Pairs (>90%)"]
    }

    # index.html’e render ile gönderiyoruz.
    return render_template('index.html', tables=all_data, summary=final_summary)


# ============================================================
# --- SERVER START ---
# ============================================================
# Flask uygulamasını başlatır. "debug=True" sayesinde,
# kodda değişiklik yaptığında otomatik yeniden başlatılır.
# ============================================================

if __name__ == '__main__':
    app.run(debug=True)
