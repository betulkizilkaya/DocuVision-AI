import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- AYARLAR ---
MODEL_PATH = "models/chess_model_v3.keras"
CLASS_INDICES_PATH = "models/class_indices.json"
BOARD_FOLDER = "models/deneme"

IMG_SIZE = (64, 64)
VALID_EXT = (".png", ".jpg", ".jpeg")

# --- MODEL + SINIFLAR ---
model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# CNN label → FEN char
CLASS_TO_FEN = {
    "White_Pawn": "P",
    "White_Rook": "R",
    "White_Knight": "N",
    "White_Bishop": "B",
    "White_Queen": "Q",
    "White_King": "K",
    "Black_Pawn": "p",
    "Black_Rook": "r",
    "Black_Knight": "n",
    "Black_Bishop": "b",
    "Black_Queen": "q",
    "Black_King": "k",
    "Empty_Square": None
}

def board_image_to_fen(image_path: str) -> str:
    img = cv2.imread(image_path)

    # --- INFERENCE UPSCALE ---
    UPSCALE = 2.5  # 2.0 – 3.0 arası dene

    img = cv2.resize(
        img,
        None,
        fx=UPSCALE,
        fy=UPSCALE,
        interpolation=cv2.INTER_CUBIC
    )

    if img is None:
        raise RuntimeError(f"Görüntü okunamadı: {image_path}")

    h, w, _ = img.shape
    cell_h = h // 8
    cell_w = w // 8

    fen_rows = []

    for row in range(8):  # rank 8 → 1
        empty_count = 0
        fen_row = ""

        for col in range(8):  # file A → H
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w

            cell = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)
            gray = gray.astype(np.float32) / 255.0
            gray = np.expand_dims(gray, axis=(0, -1))

            preds = model.predict(gray, verbose=0)[0]
            cls_name = idx_to_class[int(np.argmax(preds))]

            fen_char = CLASS_TO_FEN.get(cls_name)

            if fen_char is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += fen_char

        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    # FEN: board only (geri kalanı varsayılan)
    fen_board = "/".join(fen_rows)
    fen_full = f"{fen_board} w - - 0 1"

    return fen_full


if __name__ == "__main__":
    files = [f for f in os.listdir(BOARD_FOLDER) if f.lower().endswith(VALID_EXT)]

    if not files:
        raise RuntimeError("models/deneme içinde tahta görseli yok")

    for f in files:
        path = os.path.join(BOARD_FOLDER, f)
        fen = board_image_to_fen(path)

        print("\n==============================")
        print(f"Görsel : {f}")
        print(f"FEN    : {fen}")

