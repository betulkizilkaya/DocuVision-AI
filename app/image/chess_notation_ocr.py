from __future__ import annotations

import re
import sqlite3
import os
from typing import List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import chess

from app.core.paths import DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
temp_dir = PROJECT_ROOT / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)

def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # büyüt (çok önemli)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th

def run_tesseract_ocr(img: np.ndarray) -> str:
    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=KQRBNOabcdefgh12345678x0-+#.= "
    return pytesseract.image_to_string(img, config=config) or ""

def normalize_ocr_text(text: str) -> str:
    text = text.replace("0-0-0", "O-O-O")
    text = text.replace("0-0", "O-O")
    text = text.replace("o-o-o", "O-O-O")
    text = text.replace("o-o", "O-O")

    text = text.replace("§", "5")
    text = text.replace("|", "1")
    text = text.replace("—", "-")
    text = text.replace("–", "-")

    return text


def extract_move_candidates(text: str) -> List[str]:
    pattern1 = r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?"
    pattern2 = r"\b[a-h][1-8]\b"

    tokens = re.findall(pattern1, text)
    tokens += re.findall(pattern2, text)

    # sırayı koruyarak tekrar temizle
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    return ordered


def validate_san_sequence(tokens: List[str]) -> List[Dict[str, Any]]:
    board = chess.Board()
    rows = []

    for idx, tok in enumerate(tokens):
        is_valid = 0

        try:
            move = board.parse_san(tok)
            board.push(move)
            is_valid = 1
        except Exception:
            pass

        rows.append({
            "move_index": idx + 1,
            "raw_token": tok,
            "normalized_token": tok,
            "is_regex_match": 1,
            "is_valid_san": is_valid
        })

    return rows


def save_notation_results(
    conn: sqlite3.Connection,
    image_id: int,
    raw_text: str,
    normalized_text: str,
    filtered_text: str,
    move_rows: List[Dict[str, Any]]
) -> None:
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO notation_ocr (
            image_id, roi_type, raw_text, normalized_text, filtered_text
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        image_id,
        "full_or_roi",
        raw_text,
        normalized_text,
        filtered_text
    ))

    for row in move_rows:
        cur.execute("""
            INSERT INTO notation_moves (
                image_id, move_index, raw_token, normalized_token,
                is_regex_match, is_valid_san
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            image_id,
            row["move_index"],
            row["raw_token"],
            row["normalized_token"],
            row["is_regex_match"],
            row["is_valid_san"]
        ))

    move_count = len(move_rows)
    valid_count = sum(r["is_valid_san"] for r in move_rows)
    san_ratio = (valid_count / move_count) if move_count else 0.0

    final_label = "game_notation" if (move_count >= 5 and san_ratio >= 0.5) else "not_game_notation"

    cur.execute("""
        INSERT OR REPLACE INTO notation_summary (
            image_id, raw_text_length, move_count, valid_move_count, san_ratio, final_label, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        image_id,
        len(filtered_text),
        move_count,
        valid_count,
        san_ratio,
        final_label
    ))

    conn.commit()


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

    raw_lines = []

    for idx, line_img in enumerate(line_images):
        prep = preprocess_for_ocr(line_img)

        text = run_tesseract_ocr(prep)
        text = normalize_ocr_text(text)
        text = clean_line_prefix(text)
        text = fix_ocr_chess_errors(text)
        text = collapse_bad_square_numbers(text)
        text = clean_move_line(text)
        text = keep_chess_chars(text)
        text = split_merged_moves(text)
        text = remove_noise_words(text)

        if len(text) < 5:
            continue

        if is_move_line(text):
            raw_lines.append(text)

        ok1 = cv2.imwrite(str(temp_dir / f"line_{idx:02d}.png"), line_img)
        ok2 = cv2.imwrite(str(temp_dir / f"line_{idx:02d}_prep.png"), prep)
        print(f"line_{idx:02d}.png yazıldı mı? {ok1}")
        print(f"line_{idx:02d}_prep.png yazıldı mı? {ok2}")
        print(f"OCR line_{idx:02d}: {text}")

    raw_text = "\n".join(raw_lines)
    normalized_text = raw_text
    fixed_text = raw_text

    tokens = extract_move_candidates(fixed_text)
    tokens = [repair_move_token(t) for t in tokens]
    tokens = [t for t in tokens if is_possible_chess_move(t)]

    move_rows = validate_san_sequence(tokens)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        save_notation_results(
            conn,
            image_id,
            raw_text,
            normalized_text,
            fixed_text,
            move_rows
        )
    finally:
        conn.close()

def fix_ocr_chess_errors(text: str) -> str:
    text = text.replace("H", "B")
    text = text.replace("8", "B")
    text = text.replace("l", "1")

    # sık görülen OCR hataları
    text = text.replace("cbx", "bx")
    text = text.replace("4x", "Bx")
    text = text.replace("63", "g3")
    text = text.replace("f24", "f2#")
    text = text.replace("hh2", "bh2")

    text = re.sub(r"\bE([a-h]x[a-h][1-8])", r"e\1", text)

    return text

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

def is_move_line(text: str) -> bool:
    text = text.strip().lower()

    if not re.search(r"\d+\.", text):
        return False

    bad_words = [
        "defence", "attacking", "bagirov", "kholmov",
        "nataf", "wins", "check", "square", "from"
    ]
    if any(w in text for w in bad_words):
        return False

    move_like = re.findall(r"[kqrbn]?[a-h]?[1-8]?x?[a-h][1-8][+#]?", text)
    return len(move_like) >= 2

def clean_move_line(text: str) -> str:
    text = replace_figurines(text)

    text = re.sub(r"\b\d{4}\b", " ", text)
    text = re.sub(r"\b[01]-[01]\b", " ", text)
    text = re.sub(r"\b1/2-1/2\b", " ", text)

    text = re.sub(
        r"\b(attacking|defence|defending|wins|square|from|the|capture|on|check|bagirov|kholmov|nataf)\b",
        " ",
        text,
        flags=re.I
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text

def repair_move_token(token: str) -> str:
    token = token.strip()

    token = token.replace("E", "e")
    token = token.replace("W", "N")
    token = token.replace("8", "B")

    token = re.sub(r"[^KQRBNa-hx1-8+#=]", "", token)

    # fazla rakamı temizle
    token = re.sub(r"([a-h][1-8])[1-8]+", r"\1", token)

    if re.match(r"^[a-h]x[a-h][1-8]", token):
        return token

    if re.match(r"^[a-h][1-8]", token):
        return token

    if re.match(r"^x[a-h][1-8]", token):
        return "e" + token

    return token

def is_possible_chess_move(token: str) -> bool:
    token = token.strip()

    # en az bir kare içermeli (e4 gibi)
    if not re.search(r"[a-h][1-8]", token):
        return False

    # çok kısa ya da saçma şeyleri ele
    if len(token) < 2 or len(token) > 7:
        return False

    return True

def extract_move_region(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kontrast artır
    gray = cv2.equalizeHist(gray)

    # binary
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # yatay genişlet (satırları birleştir)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilated = cv2.dilate(th, kernel, iterations=2)

    # kontur bul
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    max_area = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # çok küçükleri ele
        if area < 5000:
            continue

        # geniş blokları tercih et (hamle sütunu)
        if area > max_area:
            max_area = area
            best = (x, y, w, h)

    if best is None:
        return img

    x, y, w, h = best
    roi = img[y:y+h, x:x+w]

    return roi

def keep_chess_chars(text: str) -> str:
    text = re.sub(r"[^KQRBNa-hxO0-9\.\-\+#= ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_line_prefix(text: str) -> str:
    text = text.strip()

    # baştaki B271..., B331..., 827 -1... gibi şeyler
    text = re.sub(r"^[A-Z]?\d+(?:\.\.\.)?", "", text)
    text = re.sub(r"^\d+\s*[-=]\s*", "", text)
    text = re.sub(r"^[A-Za-z]+\d+\.*", "", text)

    return text.strip()

def split_merged_moves(text: str) -> str:
    # harf + büyük harf → ayır
    text = re.sub(r"([a-h1-8])([KQRBN])", r"\1 \2", text)

    # sayı + harf → ayır
    text = re.sub(r"(\d)([a-h])", r"\1 \2", text)

    # hamle sonu + yeni hamle
    text = re.sub(r"([+#])([KQRBNa-h])", r"\1 \2", text)

    return text

def remove_noise_words(text: str) -> str:
    words = text.split()
    clean = []

    for w in words:
        # çok uzun ve saçma kelimeleri at
        if len(w) > 6 and not re.search(r"[1-8]", w):
            continue
        clean.append(w)

    return " ".join(clean)

def collapse_bad_square_numbers(text: str) -> str:
    # kare sonunda fazla sayı varsa tek kareye indir
    text = re.sub(r"([a-h][1-8])[1-8]+", r"\1", text)

    # taşlı hamlelerde de uygula
    text = re.sub(r"([KQRBN]?[a-h]?x?[a-h][1-8])[1-8]+", r"\1", text)

    return text

