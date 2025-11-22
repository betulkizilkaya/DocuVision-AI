import sqlite3
import io
from pathlib import Path
from PIL import Image
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

def calc_ssim(img1, img2):
    """SSIM (yapısal->parlaklık, kontrast, yapı)benzerliği [0,1]"""
    try:
        # griye çevir
        i1 = np.array(img1.convert("L"))          #ssım için image gri tonlamalı formata çevirilir ve numPy dizisine çevrilir
        i2 = np.array(img2.convert("L"))

        # boyutları eşitle
        h = min(i1.shape[0], i2.shape[0])         #boyut eşitleme için min en-boy
        w = min(i1.shape[1], i2.shape[1])
        if h < 3 or w < 3:
            return 0.0  # çok küçük resimlerde SSIM anlamsız

        i1_resized = cv2.resize(i1, (w, h))
        i2_resized = cv2.resize(i2, (w, h))

        # dinamik pencere boyutu (karşılaştırma için kayan pencere ile bölge bölge kıyaslıyoruz)
        win_size = min(7, h, w)                   #window size için maks 7 olacak şekilde en küçüğü seçilecek
        if win_size % 2 == 0:                     #merkez piksel için windowSize tek sayı olmalı
            win_size -= 1
        win_size = max(3, win_size)               #min anlamlı yapı 3x3 lük olduğu için win_size min 3 olmalı

        # güvenli SSIM hesaplama
        s, _ = ssim(i1_resized, i2_resized, full=True, win_size=win_size)     # ssim ortalama ve bölge bölge benzerlik listesi olamak üzere iki değer verir."_" ile ikincisini yok sayarız
        if np.isnan(s):  # NaN çıkarsa
            return 0.0
        return float(max(0, min(1, s)))

    except Exception as e:
        # herhangi bir hata (örnek crop, pad, shape hatası)
        return 0.0



def calc_phash(img1, img2):
    """pHash benzerliği [0,1]  """   #
    h1 = imagehash.phash(img1)
    h2 = imagehash.phash(img2)
    return 1 - (h1 - h2) / 64.0  # iki hash arasındaki hamming mesafesini(kaç bit farklı) hesaplar.Bu [0-64] aralığındaki değeri [0-1] e çeker.

def calc_orb(img1, img2):
    """ORB benzerliği [0,1]"""     #ilginç noktaları (köşeler, kenarlar) bulur ve bu noktaları tanımlar.bir görseldeki özellikleri diğeriyle eşleştirmeye çalışır.
    try:                           # Görsel döndürülmüş, ölçeklenmiş veya farklı bir açıdan çekilmiş olsa bile benzerliği tespit edebilir.
        i1 = np.array(img1.convert("L"))
        i2 = np.array(img2.convert("L"))

        # Geçersiz boyut kontrolü
        if i1.size == 0 or i2.size == 0:
            return 0.0
        if i1.shape[0] < 5 or i1.shape[1] < 5 or i2.shape[0] < 5 or i2.shape[1] < 5:
            return 0.0  # çok küçük görselleri geç

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(i1, None)        #Görseldeki "anahtar noktaları" (kp1) ve bu noktaların "tanımlayıcılarını" (des1) bulur.
        kp2, des2 = orb.detectAndCompute(i2, None)

        #Görselde tanınabilir nokta bulunmadıysa benzerlik hesaplanamaz
        if des1 is None or des2 is None:
            return 0.0

        # Brute-force eşleştirici (Hamming mesafesi)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)      #Brute-Force(Kaba Kuvvet) eşleştirici->1. görseldeki her özelliği 2. görseldeki her özelliğe karşı kontrol eder.
        matches = bf.match(des1, des2)                             #En iyi eşleşmeleri bulur.

        if not matches:
            return 0.0

        # Ortalama benzerlik
        distances = [m.distance for m in matches]
        sim = 1 - (np.mean(distances) / 256)                      #[0-1] aralıgına sokar
        return float(max(0, min(1, sim)))

    except Exception:
        return 0.0


def insert_similarity(conn, id_a, id_b, ssim_v, phash_v, orb_v, avg):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO image_similarity (image_id_a, image_id_b, ssim, phash, orb, avg_similarity)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (id_a, id_b, ssim_v, phash_v, orb_v, avg))
    conn.commit()

def process():
    conn = create_connection()
    images = fetch_images(conn)
    print(f"[INFO] {len(images)} görsel bulundu. Benzerlik hesaplanıyor...")

    for i in range(len(images)):
        id_a, blob_a = images[i]
        img_a = Image.open(io.BytesIO(blob_a))

        for j in range(i + 1, min(i + 6, len(images))):  # çok fazla kombinasyon olmasın diye 5 komşu
            id_b, blob_b = images[j]
            img_b = Image.open(io.BytesIO(blob_b))

            s = calc_ssim(img_a, img_b)
            p = calc_phash(img_a, img_b)
            o = calc_orb(img_a, img_b)
            avg = round((s + p + o) / 3, 4)

            insert_similarity(conn, id_a, id_b, s, p, o, avg)
            print(f"→ {id_a}-{id_b}: SSIM={s:.2f}, pHash={p:.2f}, ORB={o:.2f}, AVG={avg:.2f}")

    conn.close()
    print("[✓] Görsel benzerlik analizleri tamamlandı.")

if __name__ == "__main__":
    process()