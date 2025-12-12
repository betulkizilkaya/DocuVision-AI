import sqlite3
import io
import joblib
import numpy as np
from pathlib import Path
from PIL import Image

# --- YOL VE BAĞLANTI AYARLARI ---
# BASE_DIR: app/core/script/
BASE_DIR = Path(__file__).resolve().parent

# APP_DIR: app/core/
APP_DIR = BASE_DIR.parent

# ROOT_DIR: ProjectNexus-Intelligent-PDF-Analysis/
ROOT_DIR = APP_DIR.parent

# Veritabanı yolu
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"

# Model Yolu GÜNCELLENDİ: data/models/chessboard_clf.joblib
MODEL_PATH = ROOT_DIR / "data" / "models" / "doc_type_clf.joblib"


def create_connection():
    return sqlite3.connect(DB_PATH)


# --- SINIFLANDIRICI (MODEL YÜKLEME VE ÖN İŞLEME) ---
class ChessboardClassifier:
    """Satranç tahtası tespiti için modeli yükler ve tahmin yapar."""

    def __init__(self, model_path: Path):
        print(f"[MODEL] Satranç Tahtası Model yükleniyor: {model_path}")
        try:
            # Model yükleme, güncellenmiş MODEL_PATH'i kullanır
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ ML modeli bulunamadı: {model_path}. Lütfen arkadaşınızın modeli bu yola kaydettiğinden emin olun.")

    def _preprocess(self, pil_image: Image) -> np.ndarray:
        """PIL görüntüsünü modelin beklediği formata (Örn: 64x64 RGB) dönüştürür."""
        # Varsayım: Model, 64x64 RGB görüntüsünü düzleştirilmiş (flattened) bekliyor.
        img = pil_image.convert('RGB').resize((64, 64), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        # Scikit-learn modelleri için 1D diziye dönüştür
        return img_array.flatten().reshape(1, -1)

    def predict(self, blob_data: bytes):
        """BLOB verisini alır, işler ve is_chessboard, score döndürür."""
        img = Image.open(io.BytesIO(blob_data))
        preprocessed_data = self._preprocess(img)

        # Modelden tahmin al (predict_proba ile skor çekilir)
        prediction_proba = self.model.predict_proba(preprocessed_data)[0]
        score = prediction_proba[1]  # 1. sınıfın olasılığı (chessboard)
        is_chessboard = 1 if score > 0.95 else 0  # %95 eşiği

        return is_chessboard, float(score)


# --- VERİTABANI İŞLEMLERİ ---
def get_pending_images(conn: sqlite3.Connection):
    """
    Satranç tahtası bayrağı (is_chessboard) henüz NULL olan ve temel özellikleri
    (image_features.py tarafından) hesaplanmış görselleri getirir.
    """
    cur = conn.cursor()
    # image_features tablosunda is_chessboard NULL olanları çek
    query = """
            SELECT T2.image_id, T1.blob
            FROM pdf_images T1
            INNER JOIN image_features T2 ON T1.id = T2.image_id
            WHERE T2.is_chessboard IS NULL 
            """
    cur.execute(query)
    return cur.fetchall()


def update_chessboard_flag(conn: sqlite3.Connection, image_id: int, is_chessboard: int, score: float):
    """image_features tablosundaki kaydı günceller (UPDATE)."""
    cur = conn.cursor()
    cur.execute("""
                UPDATE image_features 
                SET is_chessboard = ?, chessboard_score = ?
                WHERE image_id = ?
                """, (is_chessboard, score, image_id))
    conn.commit()


# --- ANA ÇALIŞMA FONKSİYONU ---
def run_chessboard_detection():
    """Ana çalışma fonksiyonu: Modeli çalıştırır ve DB'yi günceller."""
    print("--- [GÖREV: SATRANÇ TAHTASI TESPİTİ BAŞLIYOR] ---")

    conn = create_connection()
    try:
        classifier = ChessboardClassifier(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"❌ HATA: {e}")
        conn.close()
        return

    images = get_pending_images(conn)
    print(f"[INFO] İşlenecek {len(images)} görsel bulundu.")

    for img_id, blob in images:
        try:
            is_chessboard, score = classifier.predict(blob)
            update_chessboard_flag(conn, img_id, is_chessboard, score)

            if is_chessboard:
                print(f"   → ID {img_id}: TAHTA TESPİT EDİLDİ! Skor: {score:.2f}")

        except Exception as e:
            print(f"Hata (ID: {img_id}): Görsel işlenirken hata oluştu: {e}")

    conn.close()
    print(f"--- [GÖREV: SATRANÇ TAHTASI TESPİTİ TAMAMLANDI] ---")


if __name__ == "__main__":
    run_chessboard_detection()