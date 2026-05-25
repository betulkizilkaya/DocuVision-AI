import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from app.core.paths import  ROOT_DIR



# =========================
# AYARLAR
# =========================
IMAGE_PATH = "data/iimg.png"  # Girdi görseli
MODEL_PATH = ROOT_DIR / "data" / "models" / "chessboard_cnn_v1.keras"     # Eğitilmiş CNN modelin
OUTPUT_DIR = "detected_boards"          # Kabul edilen crop'lar buraya kaydedilir

# CNN giriş boyutu
IMG_SIZE = 128

# CNN eşik değeri
CLASSIFIER_THRESHOLD = 0.65

# Contour filtreleri
MIN_AREA = 4000
MAX_AREA = 20000

# Kareye yakınlık filtresi
MIN_ASPECT_RATIO = 0.75
MAX_ASPECT_RATIO = 1.35

# İsteğe bağlı crop margin
CROP_PADDING = 4


# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def preprocess_for_classifier(crop_bgr: np.ndarray, img_size: int = 128) -> np.ndarray:
    """
    Crop'u CNN'e uygun hale getirir:
    - resize
    - normalize [0,1]
    - kanal düzenleme
    - batch boyutu ekleme
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Boş crop geldi.")

    resized = cv2.resize(crop_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    # Eğer model grayscale bekliyorsa burayı değiştir:
    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # normalized = gray.astype(np.float32) / 255.0
    # normalized = np.expand_dims(normalized, axis=-1)

    batched = np.expand_dims(normalized, axis=0)
    return batched


def predict_chessboard(model, crop_bgr: np.ndarray, img_size: int = 128) -> float:
    """
    CNN modelinden chessboard olasılığı döndürür.
    Binary classification varsayılmıştır.
    """
    x = preprocess_for_classifier(crop_bgr, img_size=img_size)
    pred = model.predict(x, verbose=0)

    # Binary sigmoid çıktı varsayımı
    # pred shape genelde (1,1) olur
    score = float(pred[0][0])
    return score


def safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 0) -> np.ndarray:
    """
    Taşma olmadan crop alır.
    """
    H, W = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return img[y1:y2, x1:x2]


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    4 noktayı sıralar:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def four_point_warp(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Eğer dört köşe ile düzgün crop almak istersen.
    Şu an opsiyonel; mevcut kodda boundingRect üzerinden crop alıyoruz.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width <= 0 or max_height <= 0:
        return None

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return warped


# =========================
# ANA İŞLEM
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    # Görsel yükle
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Görsel okunamadı: {IMAGE_PATH}")

    # Model yükle
    model = load_model(MODEL_PATH)

    # Gri ton
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kenar bulma
    canny = cv2.Canny(img_gray, 50, 150)

    # Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilation_2 = cv2.dilate(canny, kernel, iterations=1)

    # Contour bul
    board_contours, hierarchy = cv2.findContours(
        img_dilation_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sonuçları çizmek için
    board_squared = cv2.cvtColor(canny.copy(), cv2.COLOR_GRAY2BGR)

    accepted_candidates = []
    candidate_index = 0

    for contour in board_contours:
        area = cv2.contourArea(contour)

        if not (MIN_AREA < area < MAX_AREA):
            continue

        # Çokgen yaklaşımı
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Dörtgen mi?
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Kareye yakın mı?
        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            continue

        # Bounding box crop al
        crop = safe_crop(img, x, y, w, h, pad=CROP_PADDING)
        if crop is None or crop.size == 0:
            continue

        # İstersen boundingRect yerine perspective crop da deneyebilirsin:
        pts = np.array([pt[0] for pt in approx], dtype=np.float32)
        warped_crop = four_point_warp(img, pts)

        # Hangisini classifier'a vereceğini seç
        # Genelde warp daha temiz sonuç verir
        classifier_input = warped_crop if warped_crop is not None and warped_crop.size > 0 else crop

        try:
            score = predict_chessboard(model, classifier_input, img_size=IMG_SIZE)
        except Exception as e:
            print(f"[UYARI] Candidate {candidate_index} tahmin sırasında hata verdi: {e}")
            candidate_index += 1
            continue

        if score >= CLASSIFIER_THRESHOLD:
            # Kabul edilen aday
            accepted_candidates.append({
                "index": candidate_index,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "score": score
            })

            # Crop kaydet
            save_path = os.path.join(OUTPUT_DIR, f"board_{candidate_index}_score_{score:.3f}.png")
            cv2.imwrite(save_path, classifier_input)

            # Çizim
            cv2.rectangle(board_squared, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                board_squared,
                f"{score:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        else:
            # Reddedilen adayları istersen farklı renkle çiz
            cv2.rectangle(board_squared, (x, y), (x + w, y + h), (0, 0, 255), 1)

        candidate_index += 1

    print(f"Toplam kabul edilen aday sayısı: {len(accepted_candidates)}")
    for item in accepted_candidates:
        print(item)

    # Gösterim
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(board_squared, cv2.COLOR_BGR2RGB))
    plt.title("CNN ile Filtrelenmiş Dörtgen Adaylar")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()