from __future__ import annotations

import re
import sqlite3
import os
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from app.core.paths import DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
temp_dir = PROJECT_ROOT / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)

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

def normalize_ocr_text(text: str) -> str:
    text = text.replace("0-0-0", "O-O-O")
    text = text.replace("0-0", "O-O")
    text = text.replace("o-o-o", "O-O-O")
    text = text.replace("o-o", "O-O")
    text = text.replace("§", "5")
    text = text.replace("|", "1")
    text = text.replace("l", "1")
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    return text


def process_single_image(image_id: int, image_path: str) -> None:
    print("Current working directory:", os.getcwd())
    print("Dosyalar buraya yazılıyor:", temp_dir.resolve())

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")

    roi = extract_move_region(img)
    ok3 = cv2.imwrite(str(temp_dir / "debug_move_roi.png"), roi)
    print(f"debug_move_roi.png yazıldı mı? {ok3}")

    line_images = split_into_text_lines(roi)
    print("Bulunan satır sayısı:", len(line_images))

    notation_lines = extract_notation_lines(line_images)

    for idx, line in enumerate(notation_lines):
        print(f"TEXT line_{idx:02d}: {line}")

    full_text = "\n".join(notation_lines)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        save_notation_text_result(
            conn=conn,
            image_id=image_id,
            raw_text=full_text,
            normalized_text=full_text
        )
    finally:
        conn.close()

    print("[OK] Notation text extraction tamamlandı.")

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

        if cw < 80:
            continue
        if ch < 12:
            continue
        if cw * ch < 1000:
            continue

        boxes.append((x, y, cw, ch))

    boxes = sorted(boxes, key=lambda b: b[1])

    line_images = []
    for i, (x, y, cw, ch) in enumerate(boxes):
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)

        crop = img_bgr[y1:y2, x1:x2]
        line_images.append(crop)

    return line_images

def replace_figurines(text: str) -> str:
    mapping = {
        "♔": "K", "♚": "K",
        "♕": "Q", "♛": "Q",
        "♖": "R", "♜": "R",
        "♗": "B", "♝": "B",
        "♘": "N", "♞": "N",
        "♙": "",  "♟": ""
    }

    for k, v in mapping.items():
        text = text.replace(k, v)

    return text

def split_merged_moves(text: str) -> str:
    text = re.sub(r"([a-h1-8])([KQRBN])", r"\1 \2", text)
    text = re.sub(r"([+#])([KQRBNa-hO])", r"\1 \2", text)
    return text

def collapse_bad_square_numbers(text: str) -> str:
    text = re.sub(r"([a-h][1-8])[1-8]+", r"\1", text)
    text = re.sub(r"([KQRBN]?[a-h]?x?[a-h][1-8])[1-8]+", r"\1", text)
    return text

def keep_notation_text(text: str) -> str:
    text = replace_figurines(text)

    # Notasyon için gerekli temel karakterleri tut
    text = re.sub(r"[^KQRBNa-hxO0-9\.\-\+#=:/ ]", " ", text)

    # çoklu boşluk temizliği
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_move_numbers(text: str) -> str:
    # 1.e4 -> 1. e4
    text = re.sub(r"(\d+)\.([KQRBNa-hO])", r"\1. \2", text)

    # 1...e5 -> 1... e5
    text = re.sub(r"(\d+)\.\.\.([KQRBNa-hO])", r"\1... \2", text)

    # ..e5 gibi bozuk OCR kalıntılarını temizle
    text = re.sub(r"(^|\s)\.\.([KQRBNa-hO])", r"\1\2", text)

    return text

def save_notation_text_result(
    conn: sqlite3.Connection,
    image_id: int,
    raw_text: str,
    normalized_text: str
) -> None:
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO notation_ocr (
            image_id, roi_type, raw_text, normalized_text, filtered_text
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        image_id,
        "full_or_roi",
        raw_text,
        normalized_text,
        normalized_text
    ))

    conn.commit()

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

def extract_notation_lines(line_images: list[np.ndarray]) -> list[str]:
    out_lines = []

    for idx, line_img in enumerate(line_images):
        prep = preprocess_for_ocr(line_img)
        text = run_tesseract_ocr(prep)

        text = normalize_ocr_text(text)
        text = fix_common_ocr_confusions(text)
        text = collapse_bad_square_numbers(text)
        text = keep_notation_text(text)
        text = normalize_move_numbers(text)
        text = split_merged_moves(text)
        text = remove_noise_words(text)

        text = re.sub(r"\s+", " ", text).strip()

        if len(text) >= 4 and re.search(r"[a-h][1-8]", text):
            out_lines.append(text)

        cv2.imwrite(str(temp_dir / f"line_{idx:02d}.png"), line_img)
        cv2.imwrite(str(temp_dir / f"line_{idx:02d}_prep.png"), prep)

    return out_lines

def fix_common_ocr_confusions(text: str) -> str:
    text = re.sub(r"\b8(?=[a-hx])", "B", text)
    text = re.sub(r"\b8([a-h][1-8])", r"B\1", text)
    text = re.sub(r"(?<=\s)8(?=[a-h])", "B", text)

    text = re.sub(r"\b0-0-0\b", "O-O-O", text)
    text = re.sub(r"\b0-0\b", "O-O", text)

    return text

def remove_noise_words(text: str) -> str:
    noise_words = [
        "attacking", "attack", "defence", "defending",
        "check", "from", "the", "and", "wins",
        "capture", "square", "mate", "threat", "idea",
        "with", "after", "white", "black"
    ]

    for w in noise_words:
        text = re.sub(rf"\b{w}\b", " ", text, flags=re.I)

    text = re.sub(r"\s+", " ", text).strip()
    return text

