from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_IMAGES = [
    PROJECT_ROOT / "temp" / "test_notation.png",
    PROJECT_ROOT / "temp" / "test_notation_2.png",
    PROJECT_ROOT / "temp" / "test_notation_3.png",
]

OUT_DIR = PROJECT_ROOT / "temp" / "character_crop_exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def mask_board_regions(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = img_bgr.copy()
    h_img, w_img = gray.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        ratio = w / max(h, 1)

        if area < 4000:
            continue
        if not (0.75 <= ratio <= 1.25):
            continue
        if w < 60 or h < 60:
            continue
        if w > w_img * 0.6 or h > h_img * 0.6:
            continue

        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), -1)

    return out


def crop_main_text_area(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    y1 = int(h * 0.25)
    y2 = int(h * 0.96)
    return img_bgr[y1:y2, :]


def split_columns(img_bgr: np.ndarray) -> list[np.ndarray]:
    h, w = img_bgr.shape[:2]
    return [
        img_bgr[:, : w // 2],
        img_bgr[:, w // 2 :]
    ]


def resize_with_padding(img_bgr: np.ndarray, size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Boş crop")

    scale = min(size / w, size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((size, size), 255, dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def extract_character_crops(region_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)

    # Harfleri bulmak için ters binary
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    # Çok küçük parçaları biraz birleştir ama karakterleri fazla yapıştırma
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    H, W = gray.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Çok küçük noise at
        if w < 3 or h < 6:
            continue
        if area < 20:
            continue

        # Çok büyük blokları at
        if w > 80 or h > 80:
            continue

        # Sayfa kenarı/çizgi vb.
        if w > h * 4:
            continue

        boxes.append((x, y, w, h))

    # Önce yukarıdan aşağı, sonra soldan sağa sırala
    boxes = sorted(boxes, key=lambda b: (b[1] // 20, b[0]))

    crops = []

    for x, y, w, h in boxes:
        pad = 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        crop = region_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        norm = resize_with_padding(crop, size=64)
        crops.append(norm)

    return crops


def export_character_crops() -> None:
    total = 0

    for img_idx, img_path in enumerate(INPUT_IMAGES):
        if not img_path.exists():
            print(f"[WARN] Görsel yok: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Okunamadı: {img_path}")
            continue

        masked = mask_board_regions(img)
        main_area = crop_main_text_area(masked)
        columns = split_columns(main_area)

        cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_masked.png"), masked)
        cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_main_area.png"), main_area)

        for col_idx, col in enumerate(columns):
            cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_col{col_idx}.png"), col)

            crops = extract_character_crops(col)

            for crop_idx, crop in enumerate(crops):
                out_name = f"{img_path.stem}_col{col_idx}_char{crop_idx:04d}.png"
                out_path = OUT_DIR / out_name
                cv2.imwrite(str(out_path), crop)
                total += 1

            print(f"[INFO] {img_path.name} col{col_idx}: {len(crops)} karakter crop")

    print(f"[OK] Toplam karakter crop: {total}")
    print(f"[OK] Çıktı klasörü: {OUT_DIR}")


if __name__ == "__main__":
    export_character_crops()