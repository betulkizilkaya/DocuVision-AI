import sqlite3
import io
import joblib
import numpy as np
from pathlib import Path
from PIL import Image

from app.core.paths import DB_PATH, ROOT_DIR
MODEL_PATH = ROOT_DIR / "data" / "models" / "chessboard_clf.joblib"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def create_connection():
    return sqlite3.connect(str(DB_PATH))

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

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        img = pil_image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0  # ✅ EKSİK OLAN BU
        return arr.flatten().reshape(1, -1)

    def predict(self, blob_data: bytes):
        """BLOB verisini alır, işler ve is_chessboard, score döndürür."""
        img = Image.open(io.BytesIO(blob_data))
        preprocessed_data = self._preprocess(img)

        # Modelden tahmin al (predict_proba ile skor çekilir)
        prediction_proba = self.model.predict_proba(preprocessed_data)[0]
        proba = self.model.predict_proba(preprocessed_data)[0]
        classes = list(self.model.classes_)
        score = float(proba[classes.index(1)])  # label=1 chessboard

        THRESHOLD = 0.35
        is_chessboard = 1 if score >= THRESHOLD else 0

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