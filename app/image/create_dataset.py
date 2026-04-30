from __future__ import annotations

from pathlib import Path
import shutil

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "temp" / "piece_head_dataset"
OUT_DIR = PROJECT_ROOT / "temp" / "piece_head_dataset_clean"

CLASSES = ["B", "N", "R", "Q", "K", "P", "empty_or_noise"]


def extract_first_character_crop(img_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    contours, _ = cv2.findContours(
        bw,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 2 or h < 4:
            continue

        if w * h < 8:
            continue

        boxes.append((x, y, w, h))

    if not boxes:
        return None

    boxes = sorted(boxes, key=lambda b: b[0])
    x, y, w, h = boxes[0]

    pad = 3
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)

    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def resize_with_padding(img_bgr: np.ndarray, size: int = 32) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Boş görsel")

    scale = min(size / w, size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((size, size), 255, dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def prepare_output_dirs() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for cls in CLASSES:
        (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)


def build_dataset() -> None:
    prepare_output_dirs()

    total_saved = 0
    total_skipped = 0

    for cls in CLASSES:
        src_cls_dir = SOURCE_DIR / cls

        if not src_cls_dir.exists():
            print(f"[WARN] Kaynak klasör yok: {src_cls_dir}")
            continue

        files = [
            p for p in src_cls_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        ]

        print(f"[INFO] {cls}: kaynak görsel sayısı = {len(files)}")

        saved_for_cls = 0

        for idx, img_path in enumerate(files):
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"[WARN] Okunamadı: {img_path}")
                total_skipped += 1
                continue

            char_crop = extract_first_character_crop(img)

            if char_crop is None:
                print(f"[WARN] İlk karakter bulunamadı: {img_path.name}")
                total_skipped += 1
                continue

            try:
                normalized = resize_with_padding(char_crop, size=32)
            except Exception as e:
                print(f"[WARN] Normalize edilemedi: {img_path.name} -> {e}")
                total_skipped += 1
                continue

            out_name = f"{cls}_{idx:04d}_{img_path.stem}.png"
            out_path = OUT_DIR / cls / out_name

            cv2.imwrite(str(out_path), normalized)

            saved_for_cls += 1
            total_saved += 1

        print(f"[OK] {cls}: kaydedilen = {saved_for_cls}")

    print("\n[DONE] Temiz dataset oluşturuldu.")
    print(f"[OK] Toplam kaydedilen: {total_saved}")
    print(f"[OK] Atlanan: {total_skipped}")
    print(f"[OK] Çıktı klasörü: {OUT_DIR}")


if __name__ == "__main__":
    build_dataset()