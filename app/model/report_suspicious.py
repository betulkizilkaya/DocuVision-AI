# report_suspicious.py (DB'ye yazmaz; OUT_JSON istediğin gibi)

import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---- PATHS (senin verdiğin) ----
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent.parent

IMAGES_DIR = ROOT_DIR / "data" / "chessboard_dataset" / "chessboard"

OUT_JSON = Path("reports") / "suspicious_report.json"  # istediğin format

MODEL_PATH = Path("models")  / "chess_model_v3.keras"
CLASS_INDICES_PATH = Path("models") / "class_indices.json"

# ---- AYARLAR ----
IMG_SIZE = (64, 64)
VALID_EXT = (".png", ".jpg", ".jpeg")
UPSCALE = 2.5

CONF_TH = 0.60
MARGIN_TH = 0.15
MAX_SQ_SAMPLES = 25

CLASS_TO_FEN = {
    "White_Pawn": "P", "White_Rook": "R", "White_Knight": "N", "White_Bishop": "B",
    "White_Queen": "Q", "White_King": "K",
    "Black_Pawn": "p", "Black_Rook": "r", "Black_Knight": "n", "Black_Bishop": "b",
    "Black_Queen": "q", "Black_King": "k",
    "Empty_Square": None
}

def infer_suspicious_squares(image_path: Path, model, idx_to_class: dict):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Görüntü okunamadı: {image_path}")

    img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    h, w, _ = img.shape
    cell_h = h // 8
    cell_w = w // 8

    suspicious = []

    for row in range(8):
        for col in range(8):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w

            cell = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)
            gray = gray.astype(np.float32) / 255.0
            gray = np.expand_dims(gray, axis=(0, -1))

            preds = model.predict(gray, verbose=0)[0].astype(np.float64)
            order = np.argsort(preds)
            top1 = int(order[-1])
            top2 = int(order[-2])

            conf1 = float(preds[top1])
            conf2 = float(preds[top2])
            margin = conf1 - conf2

            cls1 = idx_to_class[top1]
            cls2 = idx_to_class[top2]
            fen_char = CLASS_TO_FEN.get(cls1)

            reasons = []
            if conf1 < CONF_TH:
                reasons.append(f"conf<{CONF_TH}")
            if margin < MARGIN_TH:
                reasons.append(f"margin<{MARGIN_TH}")

            if reasons:
                square = f"{chr(ord('a') + col)}{8 - row}"
                suspicious.append({
                    "square": square,
                    "cls1": cls1,
                    "cls2": cls2,
                    "fen_char": fen_char,
                    "conf1": conf1,
                    "conf2": conf2,
                    "margin": margin,
                    "reasons": reasons
                })

    suspicious.sort(key=lambda x: (x["conf1"], x["margin"]))
    return suspicious[:MAX_SQ_SAMPLES], len(suspicious)

def main():
    if not IMAGES_DIR.exists():
        raise RuntimeError(f"IMAGES_DIR yok: {IMAGES_DIR}")

    files = [p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in VALID_EXT]
    if not files:
        raise RuntimeError(f"Görsel yok: {IMAGES_DIR}")

    model = load_model(str(MODEL_PATH))
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}

    report = {
        "images_dir": str(IMAGES_DIR),
        "model": str(MODEL_PATH),
        "upscale": UPSCALE,
        "conf_th": CONF_TH,
        "margin_th": MARGIN_TH,
        "items": []
    }

    for path in files:
        sample, total = infer_suspicious_squares(path, model, idx_to_class)
        report["items"].append({
            "file": path.name,
            "path": str(path.resolve()),
            "suspicious_square_count": total,
            "suspicious_squares_sample": sample
        })
        print(f"{path.name} | suspicious_squares={total}")

    # OUT_JSON'i proje köküne göre yazmak için:
    out_path = ROOT_DIR / OUT_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport: {out_path}")

if __name__ == "__main__":
    main()
