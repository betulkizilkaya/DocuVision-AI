from __future__ import annotations

import re
import os
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# 🔥 TESSERACT FIX
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
temp_dir = PROJECT_ROOT / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------
# OCR PREPROCESS
# ---------------------------
def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 11
    )
    return th


def run_tesseract_ocr(img: np.ndarray) -> str:
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(img, config=config) or ""


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def process_single_image(image_id: int, image_path: str) -> str:
    print("Current working directory:", os.getcwd())
    print("Temp path:", temp_dir.resolve())

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")

    roi = extract_move_region(img)
    cv2.imwrite(str(temp_dir / "debug_move_roi.png"), roi)

    line_images = split_into_text_lines(roi)
    print("Bulunan satır sayısı:", len(line_images))

    notation_lines = extract_notation_lines(line_images)

    for idx, line in enumerate(notation_lines):
        print(f"TEXT line_{idx:02d}: {line}")

    full_text = "\n".join(notation_lines)

    print("[OK] OCR tamamlandı")
    return full_text   # 🔥 ARTIK DB YOK → LOCK YOK


# ---------------------------
# REGION DETECTION
# ---------------------------
def extract_move_region(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilated = cv2.dilate(th, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    max_area = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 5000:
            continue
        if area > max_area:
            max_area = area
            best = (x, y, w, h)

    if best is None:
        return img

    x, y, w, h = best
    return img[y:y+h, x:x+w]


# ---------------------------
# LINE SPLIT
# ---------------------------
def split_into_text_lines(img_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    boxes = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw < 80 or ch < 12 or cw * ch < 1000:
            continue

        boxes.append((x, y, cw, ch))

    boxes = sorted(boxes, key=lambda b: b[1])

    line_images = []
    for (x, y, cw, ch) in boxes:
        pad = 5
        crop = img_bgr[
            max(0, y-pad):min(h, y+ch+pad),
            max(0, x-pad):min(w, x+cw+pad)
        ]
        line_images.append(crop)

    return line_images


# ---------------------------
# CLEANING
# ---------------------------
def extract_notation_lines(line_images: list[np.ndarray]) -> list[str]:
    out_lines = []

    for idx, line_img in enumerate(line_images):
        prep = preprocess_for_ocr(line_img)
        text = run_tesseract_ocr(prep)

        text = re.sub(r"[^KQRBNa-hxO0-9\.\-\+#= ]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) >= 4 and re.search(r"[a-h][1-8]", text):
            out_lines.append(text)

        cv2.imwrite(str(temp_dir / f"line_{idx:02d}.png"), line_img)
        cv2.imwrite(str(temp_dir / f"line_{idx:02d}_prep.png"), prep)

    return out_lines