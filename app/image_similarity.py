import sqlite3
import io
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import imagehash

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"


def create_connection():
    return sqlite3.connect(DB_PATH)


def fetch_images(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, blob FROM pdf_images")
    return cur.fetchall()


def preprocess_image(img_pil):
    """Görselleri griye ve numpy dizisine çevirir."""
    img_gray = img_pil.convert("L")
    return np.array(img_gray)


def calc_ssim(img1, img2):
    """
    SSIM (Negatif Kontrollü):
    Siyah zemin/Beyaz yazı durumunu yakalamak için tersini de kontrol eder.
    """
    try:
        i1 = preprocess_image(img1)
        i2 = preprocess_image(img2)

        h = min(i1.shape[0], i2.shape[0])
        w = min(i1.shape[1], i2.shape[1])
        if h < 7 or w < 7: return 0.0

        i1 = cv2.resize(i1, (w, h))
        i2 = cv2.resize(i2, (w, h))

        win_size = min(7, h, w)
        if win_size % 2 == 0: win_size -= 1

        # 1. Normal Kıyaslama
        score_normal, _ = ssim(i1, i2, full=True, win_size=win_size)

        # 2. Negatif (Inverted) Kıyaslama
        i2_inverted = 255 - i2
        score_inverted, _ = ssim(i1, i2_inverted, full=True, win_size=win_size)

        s_final = max(
            0.0 if np.isnan(score_normal) else score_normal,
            0.0 if np.isnan(score_inverted) else score_inverted
        )
        return float(max(0, min(1, s_final)))
    except Exception:
        return 0.0


def calc_phash(img1, img2):
    """pHash benzerliği [0,1]"""
    try:
        h1 = imagehash.phash(img1)
        h2 = imagehash.phash(img2)
        return 1 - (h1 - h2) / 64.0
    except Exception:
        return 0.0


def calc_orb(img1, img2):
    """ORB (Geliştirilmiş): CLAHE ile detay artırma"""
    try:
        i1 = preprocess_image(img1)
        i2 = preprocess_image(img2)
        if min(i1.shape) < 10 or min(i2.shape) < 10: return 0.0

        # Kontrast Eşitleme (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        i1 = clahe.apply(i1)
        i2 = clahe.apply(i2)

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(i1, None)
        kp2, des2 = orb.detectAndCompute(i2, None)

        if des1 is None or des2 is None: return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches: return 0.0

        # En iyi %15 eşleşme
        matches = sorted(matches, key=lambda x: x.distance)
        top_matches = matches[:max(1, int(len(matches) * 0.15))]

        distances = [m.distance for m in top_matches]
        if not distances: return 0.0

        sim = 1 - (np.mean(distances) / 100.0)
        return float(max(0, min(1, sim)))
    except Exception:
        return 0.0


def calc_akaze(img1, img2):
    """
    AKAZE (Accelerated-KAZE):
    Hızlı ve detaylı özellik eşleştirme algoritması.
    """
    try:
        i1 = preprocess_image(img1)
        i2 = preprocess_image(img2)
        if min(i1.shape) < 10 or min(i2.shape) < 10: return 0.0

        akaze = cv2.AKAZE_create()

        kp1, des1 = akaze.detectAndCompute(i1, None)
        kp2, des2 = akaze.detectAndCompute(i2, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        # Lowe's Ratio Test
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(kp1) == 0 or len(kp2) == 0: return 0.0

        score = len(good_matches) / min(len(kp1), len(kp2))
        return float(min(1.0, score))

    except Exception as e:
        print(f"AKAZE Hatası: {e}")
        return 0.0


def insert_similarity(conn, id_a, id_b, ssim_v, phash_v, orb_v, akaze_v, avg):
    cur = conn.cursor()
    cur.execute("""
                INSERT INTO image_similarity (image_id_a, image_id_b, ssim, phash, orb, akaze, avg_similarity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (id_a, id_b, ssim_v, phash_v, orb_v, akaze_v, avg))
    conn.commit()


def process():
    conn = create_connection()
    images = fetch_images(conn)
    print(f"[INFO] {len(images)} görsel bulundu. (SSIM + pHash + ORB + AKAZE) Analizi başlıyor...")

    for i in range(len(images)):
        id_a, blob_a = images[i]
        img_a = Image.open(io.BytesIO(blob_a))

        for j in range(i + 1, min(i + 6, len(images))):
            id_b, blob_b = images[j]
            img_b = Image.open(io.BytesIO(blob_b))

            # Hesaplamalar
            s = calc_ssim(img_a, img_b)
            p = calc_phash(img_a, img_b)
            o = calc_orb(img_a, img_b)
            a = calc_akaze(img_a, img_b)  # AKAZE

            # 0 olmayan skorların ortalaması
            valid_scores = [val for val in [s, p, o, a] if val > 0]
            avg = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

            insert_similarity(conn, id_a, id_b, s, p, o, a, avg)
            print(f"→ {id_a}-{id_b}: SSIM={s:.2f} | pHash={p:.2f} | ORB={o:.2f} | AKAZE={a:.2f} || AVG={avg:.2f}")

    conn.close()
    print("[✓] Analiz tamamlandı.")


if __name__ == "__main__":
    process()