from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
import joblib
import chess

import cv2
import numpy as np
import pytesseract

from app.core.paths import DB_PATH

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

temp_dir = PROJECT_ROOT / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "piece_head_clf.joblib"

if MODEL_PATH.exists():
    PIECE_MODEL = joblib.load(MODEL_PATH)
    print("[OK] Piece classifier yüklendi")
else:
    PIECE_MODEL = None
    print("[WARN] Model bulunamadı")

PIECE_CLASSES = ["B", "N", "R", "Q", "K", "P", "empty_or_noise"]

def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return gray

def run_tesseract_ocr(img: np.ndarray) -> str:
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(img, config=config) or ""


def detect_board_regions(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    boxes = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        ratio = bw / max(bh, 1)

        if area < 2000:
            continue

        if not (0.6 <= ratio <= 1.4):
            continue

        if bw < 40 or bh < 40:
            continue

        boxes.append((x, y, bw, bh))

    return boxes


def mask_board_regions(img_bgr: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    boxes = detect_board_regions(img_bgr)

    for i, (x, y, w, h) in enumerate(boxes):
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(out.shape[1], x + w + pad)
        y2 = min(out.shape[0], y + h + pad)

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), -1)

    print(f"[INFO] Maskelenen tahta sayısı: {len(boxes)}")
    return out


def crop_main_text_area(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    # Başlık ve üstteki problem diyagramlarını azaltmak için.
    # Gerekirse 0.25 / 0.30 / 0.35 dene.
    y1 = int(h * 0.25)
    y2 = int(h * 0.96)

    return img_bgr[y1:y2, :]


def split_into_problem_blocks(img_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    # Daha yüksek kernel: aynı problem bloğundaki satırları birleştirir
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 18))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    boxes = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw < 80:
            continue
        if ch < 25:
            continue
        if cw * ch < 2500:
            continue

        boxes.append((x, y, cw, ch))

    boxes = sorted(boxes, key=lambda b: b[1])

    blocks = []
    for x, y, cw, ch in boxes:
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            blocks.append(crop)

    return blocks


def replace_figurines(text: str) -> str:
    mapping = {
        "♔": "K", "♚": "K",
        "♕": "Q", "♛": "Q",
        "♖": "R", "♜": "R",
        "♗": "B", "♝": "B",
        "♘": "N", "♞": "N",
        "♙": "", "♟": "",
    }

    for k, v in mapping.items():
        text = text.replace(k, v)

    return text


def normalize_ocr_text(text: str) -> str:
    text = replace_figurines(text)

    text = text.replace("0-0-0", "O-O-O")
    text = text.replace("0-0", "O-O")
    text = text.replace("o-o-o", "O-O-O")
    text = text.replace("o-o", "O-O")

    text = text.replace("|", "1")
    text = re.sub(r"\b[lI]\b", "1", text)

    text = text.replace("—", "-")
    text = text.replace("–", "-")

    return text


def fix_common_ocr_confusions(text: str) -> str:
    text = normalize_ocr_text(text)

    text = re.sub(r"\b8(?=[a-hx])", "B", text)
    text = re.sub(r"\b9(?=[a-hx])", "B", text)
    text = re.sub(r"\b6(?=[a-hx])", "B", text)

    return text

def keep_notation_text(text: str) -> str:
    # Satranç notasyonu için gerekli karakterleri koru
    text = re.sub(r"[^KQRBNOa-hx0-9\.\-\+#=:/ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_move_numbers(text: str) -> str:
    text = re.sub(r"(\d+)\.([KQRBNa-hO])", r"\1. \2", text)
    text = re.sub(r"(\d+)\.\.\.([KQRBNa-hO])", r"\1... \2", text)
    return text


def remove_noise_words(text: str) -> str:
    noise_words = [
        "practical", "examples",
        "black", "white",
        "wins", "drawn",
        "the", "and", "from", "with", "after",
        "attack", "attacking",
        "defence", "defending",
        "check", "mate",
    ]

    for w in noise_words:
        text = re.sub(rf"\b{w}\b", " ", text, flags=re.I)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    text = normalize_ocr_text(text)
    text = fix_common_ocr_confusions(text)
    text = keep_notation_text(text)
    text = normalize_move_numbers(text)
    text = remove_noise_words(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_header_before_moves(text: str) -> str:
    # Önce "1. e4", "1. d4" gibi gerçek başlangıçları ara
    m = re.search(r"\b1\.{1,3}\s*[a-hKQRBNO]", text)
    if m:
        return text[m.start():].strip()

    # OCR "1 e4" veya "1 4" gibi bozduysa
    patterns = [
        r"\b1\s+[a-h]\d?\s+",
        r"\b1\s+[a-h]\s+",
        r"\b1\s+\d\s+",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return text[m.start():].strip()

    return text

def is_notation_block(text: str) -> bool:
    if len(text) < 3:
        return False

    # 1141. gibi problem numarası
    if re.search(r"\b\d{3,4}\.", text):
        return True

    # 1. e4 tarzı hamle numarası
    if re.search(r"\b\d{1,3}\.", text) and re.search(r"[a-h][1-8]", text):
        return True

    # En az iki satranç karesi varsa notasyon olma ihtimali yüksek
    squares = re.findall(r"[a-h][1-8]", text)
    if len(squares) >= 2:
        return True

    # Rok
    if "O-O" in text:
        return True

    return False

def split_columns(img_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    middle_x1 = int(w * 0.42)
    middle_x2 = int(w * 0.58)
    middle_strip = gray[:, middle_x1:middle_x2]

    white_ratio = np.mean(middle_strip > 230)
    print(f"[INFO] Column white ratio: {white_ratio:.2f}")

    # Bu sayfa tipi için daha esnek eşik
    if white_ratio > 0.55:
        left = img_bgr[:, : w // 2]
        right = img_bgr[:, w // 2 :]
        print("[INFO] 2 sütun algılandı")
        return [left, right]

    print("[INFO] Tek sütun algılandı")
    return [img_bgr]

def extract_notation_lines_from_page(img_bgr: np.ndarray) -> list[str]:
    masked = mask_board_regions(img_bgr)
    cv2.imwrite(str(temp_dir / "debug_masked_boards.png"), masked)

    main_area = crop_main_text_area(masked)
    cv2.imwrite(str(temp_dir / "debug_main_text_area.png"), main_area)

    columns = split_columns(main_area)

    candidates: list[tuple[int, str]] = []

    for col_idx, col_img in enumerate(columns):
        cv2.imwrite(str(temp_dir / f"debug_col_{col_idx}.png"), col_img)

        line_imgs = split_into_problem_blocks(col_img)
        print(f"[INFO] Column {col_idx} problem bloğu sayısı: {len(line_imgs)}")

        for line_idx, line_img in enumerate(line_imgs):
            raw_preview = run_tesseract_ocr(preprocess_for_ocr(line_img))
            preview_text = clean_text(raw_preview)

            # eğer hamle içermiyorsa skip et
            if not re.search(r"\b1\.?\s*[a-hKQRBN0-9]", preview_text):
                continue

            move_img = line_img
            cv2.imwrite(str(temp_dir / f"col{col_idx}_block{line_idx:02d}_move_area.png"), move_img)

            prep = preprocess_for_ocr(move_img)

            # 1) Blok OCR
            raw = run_tesseract_ocr(prep)
            text_block = clean_text(raw)
            text_block = remove_header_before_moves(text_block)
            text_block = fix_chess_moves(text_block)
            #text_block = validate_and_fix_moves(text_block)
            text_block = remove_next_problem_tail(text_block)

            # 2) Şimdilik token/model OCR kapalı
            text_tokens = ""
            text = text_block

            print(f"BLOCK OCR {col_idx}-{line_idx:02d}: {text_block}")
            print(f"TOKEN OCR {col_idx}-{line_idx:02d}: {text_tokens}")
            print(f"CHOSEN    {col_idx}-{line_idx:02d}: {text}")

            cv2.imwrite(str(temp_dir / f"col{col_idx}_block{line_idx:02d}.png"), line_img)
            cv2.imwrite(str(temp_dir / f"col{col_idx}_block{line_idx:02d}_prep.png"), prep)

            if is_notation_block(text):
                s1 = score_notation_text(text)
                s2 = score_with_chess_engine(text)

                s = s1 + (s2 * 5)
                print(f"SCORE     {col_idx}-{line_idx:02d}: {s}")

                # çok bozuk OCR bloklarını ele
                if s >= 70:
                    candidates.append((s, text))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text = candidates[0]
        print(f"[BEST] score={best_score}: {best_text}")
        return [best_text]

    return []


def save_notation_text_result(
    conn: sqlite3.Connection,
    image_id: int,
    raw_text: str,
    normalized_text: str,
) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO notation_ocr (
            image_id, roi_type, raw_text, normalized_text, filtered_text
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            image_id,
            "board_masked_column_layout",
            raw_text,
            normalized_text,
            normalized_text,
        ),
    )

    conn.commit()


def process_single_image(image_id: int, image_path: str) -> None:
    print("Current working directory:", os.getcwd())
    print("Dosyalar buraya yazılıyor:", temp_dir.resolve())

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")

    notation_lines = extract_notation_lines_from_page(img)

    for idx, line in enumerate(notation_lines):
        print(f"TEXT line_{idx:02d}: {line}")

    full_text = "\n".join(notation_lines)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        save_notation_text_result(
            conn=conn,
            image_id=image_id,
            raw_text=full_text,
            normalized_text=full_text,
        )
    finally:
        conn.close()

    print("[OK] Notation extraction tamamlandı.")

def segment_block_tokens(block_img: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    # Tokenları ayırmak için hafif yatay birleştirme
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    merged = cv2.dilate(bw, kernel, iterations=1)

    contours, _ = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = gray.shape
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 5 or h < 8:
            continue

        if w * h < 40:
            continue

        if w > W * 0.7 or h > H * 0.5:
            continue

        boxes.append((x, y, w, h))

    # satır sırası + soldan sağa
    boxes = sorted(boxes, key=lambda b: (b[1] // 25, b[0]))

    tokens = []

    for x, y, w, h in boxes:
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        crop = block_img[y1:y2, x1:x2]

        if crop.size > 0:
            tokens.append(crop)

    return tokens


def extract_first_char_for_model(token_img: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(token_img, cv2.COLOR_BGR2GRAY)

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
    x2 = min(token_img.shape[1], x + w + pad)
    y2 = min(token_img.shape[0], y + h + pad)

    crop = token_img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def run_tesseract_token_ocr(img: np.ndarray) -> str:
    prep = preprocess_for_ocr(img)
    config = "--oem 3 --psm 8"
    return pytesseract.image_to_string(prep, config=config) or ""


def clean_token_text(text: str) -> str:
    text = clean_text(text)
    text = text.replace(" ", "")
    return text.strip()


def reconstruct_block_with_piece_model(block_img: np.ndarray, block_idx: str = "") -> str:
    # Önce block OCR ile hamle başlangıcı var mı kontrol et
    prep = preprocess_for_ocr(block_img)
    raw_block = run_tesseract_ocr(prep)
    clean_block = clean_text(raw_block)

    # Eğer hamle başlangıcı yoksa token/model çalıştırma
    if not re.search(r"\b1\.?\s*[a-hKQRBN0-9]", clean_block):
        return ""

    tokens = segment_block_tokens(block_img)
    out_tokens = []

    for tok_idx, tok_img in enumerate(tokens):
        raw_tok = run_tesseract_token_ocr(tok_img)
        tok_text = clean_token_text(raw_tok)

        # Başlık/gürültü tokenlarını at
        if not tok_text:
            continue

        if re.search(r"[A-Z]{2,}", tok_text):
            continue

        if re.search(r"\d{4}", tok_text):
            continue

        if len(tok_text) > 8:
            continue

        first_char = extract_first_char_for_model(tok_img)

        piece = None
        if first_char is not None:
            piece = predict_piece_from_crop(first_char)

            cv2.imwrite(
                str(temp_dir / f"debug_piece_{block_idx}_tok{tok_idx:03d}.png"),
                first_char,
            )

        if piece in ["B", "N", "R", "Q", "K"]:
            if tok_text and not re.match(r"^[KQRBN]", tok_text):
                if re.search(r"[a-hx]", tok_text):
                    tok_text = piece + tok_text

        if tok_text:
            out_tokens.append(tok_text)

        cv2.imwrite(
            str(temp_dir / f"debug_token_{block_idx}_tok{tok_idx:03d}.png"),
            tok_img,
        )

    return " ".join(out_tokens)

def predict_piece_from_crop(img_bgr: np.ndarray) -> str | None:
    if PIECE_MODEL is None:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if h < 5 or w < 5:
        return None

    # normalize 32x32
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_CUBIC)
    arr = resized.astype(np.float32) / 255.0
    arr = arr.flatten().reshape(1, -1)

    pred = PIECE_MODEL.predict(arr)[0]

    if pred in ["B", "N", "R", "Q", "K"]:
        return pred

    return None

def fix_chess_moves(text: str) -> str:
    # Eksik nokta/düzensiz hamle numarası düzeltmeleri
    text = re.sub(r"\b(\d{1,2})\s+([KQRBN]?[a-h]?[x]?[a-h][1-8])\b", r"\1. \2", text)

    # Sık görülen taş harfi düşmeleri
    text = re.sub(r"\b2d3\b", "Bd3", text)
    text = re.sub(r"\b2d\b", "Nd", text)
    text = re.sub(r"\b2e7\b", "Ne7", text)
    text = re.sub(r"\b2b7\b", "Nb7", text)

    # c6 bazen sadece 6 okunuyor; sadece 1. e4 sonrası düzelt
    text = re.sub(r"\b1\. e4 6\b", "1. e4 c6", text)

    # dxe4 parçalanınca
    text = text.replace("dxe 4", "dxe4")
    text = text.replace("dxe 4.", "dxe4")
    text = text.replace("ded4", "dxe4")

    text = re.sub(r"\s+", " ", text).strip()

    # Başlangıç hamlesi
    text = re.sub(r"\b1\s+4\b", "1. e4", text)
    text = re.sub(r"\b1\s+e4\b", "1. e4", text)
    text = re.sub(r"\b1e4\b", "1. e4", text)

    # Hamle numarası boşluk/nokta düzeltme
    text = re.sub(r"\b(\d{1,2})\.([KQRBNOa-h])", r"\1. \2", text)
    text = re.sub(r"\b(\d{1,2})\s+([a-h][1-8])\b", r"\1. \2", text)

    # c6 OCR'da 06 geliyor
    text = re.sub(r"\b06\b", "c6", text)

    # Tek haneli/yarım piyon hamlelerini çok sınırlı düzelt
    text = re.sub(r"\bd\b(?=\s+\d+\.)", "d5", text)
    text = re.sub(r"\bg\b(?=\s+\d+\.)", "g6", text)

    # 23 -> Nc3 ama hamle numarası 23 olmasın
    text = re.sub(r"(?<![\d.])\b23\b(?!\.)", "Nc3", text)

    # Rok
    text = re.sub(r"\b0-0-0\b", "O-O-O", text)
    text = re.sub(r"\b0-0\b", "O-O", text)
    text = re.sub(r"\bo-o-o\b", "O-O-O", text, flags=re.I)
    text = re.sub(r"\bo-o\b", "O-O", text, flags=re.I)

    # Capture bozulmaları
    text = re.sub(r"\bed\b", "exd", text)
    text = re.sub(r"\bde\b", "dxe", text)
    text = re.sub(r"\bded\b", "dxe", text)

    # Qe6+- -> Qe6+
    text = re.sub(
        r"([KQRBN]?[a-h]?[x]?[a-h][1-8](?:=[QRBN])?)[+][\-]+",
        r"\1+",
        text
    )

    # Sonuç düzeltme
    text = text.replace("1:0", "1-0")
    text = text.replace("0:1", "0-1")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_move_lines(text: str) -> list[str]:
    lines = text.split("\n")

    move_lines = []

    for line in lines:
        # sadece "1. e4" gibi başlayan satırlar
        if re.search(r"\b1\.\s*[a-hKQRNB]", line):
            move_lines.append(line)

    return move_lines

def remove_next_problem_tail(text: str) -> str:
    m = re.search(r"\s\d{3,4}\.\s", text)
    if m:
        return text[:m.start()].strip()
    return text


def score_notation_text(text: str) -> int:
    score = 0

    # düzgün hamle numaraları
    score += len(re.findall(r"\b\d{1,2}\.", text)) * 3

    # düzgün kareler
    score += len(re.findall(r"[a-h][1-8]", text)) * 2

    # taşlı hamleler
    score += len(re.findall(r"\b[KQRBN][a-hx]?[a-h][1-8]", text)) * 3

    # rok
    score += len(re.findall(r"O-O(?:-O)?", text)) * 3

    # sonuç
    score += len(re.findall(r"\b(?:1-0|0-1|1/2-1/2)\b", text)) * 2

    # gürültü cezaları
    score -= len(re.findall(r"\b[ae]{2,}\b", text)) * 4
    score -= len(re.findall(r"\b[a-h]\b", text)) * 2
    score -= len(re.findall(r"\b\d\b", text)) * 2

    return score

def score_with_chess_engine(text: str) -> int:
    board = chess.Board()
    tokens = text.split()

    legal_count = 0

    for tok in tokens:
        if re.match(r"\d+\.", tok):
            continue

        try:
            move = board.parse_san(tok)
            board.push(move)
            legal_count += 1
        except:
            continue

    return legal_count