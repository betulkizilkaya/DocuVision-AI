from pathlib import Path
import re

import cv2
import numpy as np
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMP_DIR = PROJECT_ROOT / "temp"

INPUT_BLOCKS = sorted(
    p for p in TEMP_DIR.glob("col*_block*.png")
    if "_prep" not in p.stem
)

OUT_DIR = TEMP_DIR / "move_token_exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_board_bbox(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h
        aspect = w / max(float(h), 1.0)

        if area < 5000:
            continue

        if 0.6 < aspect < 1.4 and area > best_area:
            best = (x, y, w, h)
            best_area = area

    return best


def remove_header_from_top(top_area: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(top_area, cv2.COLOR_BGR2GRAY)

    row_density = np.mean(gray < 200, axis=1)
    h = len(row_density)

    threshold = 0.08
    start = 0

    for i in range(int(h * 0.2), h):
        if row_density[i] > threshold:
            start = i
            break

    return top_area[start:h, :]


def crop_move_area(block_img: np.ndarray) -> np.ndarray:
    board = find_board_bbox(block_img)

    H, W = block_img.shape[:2]

    if board is None:
        return block_img[int(H * 0.4):H, :]

    x, y, w, h = board

    top_area = block_img[0:y, :]

    th, _ = top_area.shape[:2]

    # En iyi çalışan sürüm: tahta üstündeki alanın alt kısmı
    return top_area[int(th * 0.6):th, :]


def extract_tokens(img_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
    merged = cv2.dilate(bw, kernel, iterations=1)

    contours, _ = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    H, W = gray.shape[:2]
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 5 or h < 8:
            continue
        if w * h < 35:
            continue
        if w > W * 0.55:
            continue
        if h > H * 0.25:
            continue

        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1] // 25, b[0]))

    crops = []

    for x, y, w, h in boxes:
        pad = 2

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        crop = img_bgr[y1:y2, x1:x2]

        if crop.size > 0:
            crops.append(crop)

    return crops


def ocr_token(tok_img: np.ndarray) -> str:
    gray = cv2.cvtColor(tok_img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(
        gray,
        config="--psm 7 -c tessedit_char_whitelist=KQRBNabcdefgh12345678xO-+=#.",
    )

    text = text.strip()
    text = text.replace("0-0", "O-O")
    text = text.replace("o-o", "O-O")
    text = text.replace(" ", "")

    return text


def is_chess_token(text: str) -> bool:
    t = text.strip()

    if not t:
        return False

    # çok uzun = başlık
    if len(t) > 7:
        return False

    # tamamen büyük harf = isim
    if t.isupper() and len(t) > 2:
        return False

    # yıl
    if re.match(r"^\d{3,4}$", t):
        return False

    # hamle numarası
    if re.match(r"^\d+(\.\.\.|\.)$", t):
        return True

    # sonuç
    if t in {"1-0", "0-1", "1/2-1/2"}:
        return True

    # rok
    if t in {"O-O", "O-O-O"}:
        return True

    # normal hamle
    if re.match(r"^[KQRBN]?[a-h][1-8](=[QRBN])?[+#]?$", t):
        return True

    # capture
    if re.match(r"^[KQRBN]?[a-h]?x[a-h][1-8](=[QRBN])?[+#]?$", t):
        return True

    return False


def clear_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in OUT_DIR.glob("*"):
        if p.is_file():
            p.unlink()


def main() -> None:
    clear_output_dir()

    if not INPUT_BLOCKS:
        print("[WARN] temp içinde col*_block*.png bulunamadı.")
        print("Önce run_chess_notation_ocr.py çalıştır.")
        return

    total = 0

    for block_idx, block_path in enumerate(INPUT_BLOCKS):
        img = cv2.imread(str(block_path))

        if img is None:
            print(f"[WARN] Okunamadı: {block_path}")
            continue

        move_area = crop_move_area(img)

        move_area_path = OUT_DIR / f"{block_path.stem}_move_area.png"
        cv2.imwrite(str(move_area_path), move_area)

        tokens = extract_tokens(move_area)

        saved_for_block = 0

        for i, tok in enumerate(tokens):
            token_text = ocr_token(tok)

            if not is_chess_token(token_text):
                continue

            out_path = OUT_DIR / f"{block_path.stem}_tok_{i:03d}_{token_text}.png"
            cv2.imwrite(str(out_path), tok)

            print(f"[TOKEN] {block_path.stem} | {token_text}")

            saved_for_block += 1
            total += 1

        print(f"[OK] {block_path.name}: kaydedilen token={saved_for_block}")

    print(f"[DONE] Toplam token crop: {total}")
    print(f"[OK] Çıktı klasörü: {OUT_DIR}")


if __name__ == "__main__":
    main()