import os
import cv2
import json
import numpy as np
import shutil
from tensorflow.keras.models import load_model

# === PATHLER ===
MODEL_PATH = "models/chess_model_v3.keras"
CLASS_INDICES_PATH = "models/class_indices.json"

INPUT_ROOT = "extracted_squares"
OUTPUT_ROOT = "sorted_pieces"

IMG_SIZE = (64, 64)

# === MODEL YÜKLE ===
model = load_model(MODEL_PATH)

# class_indices ters çevir (index -> class_name)
with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

print("[INFO] Sınıflar:", idx_to_class)

# === OUTPUT klasörlerini oluştur ===
for class_name in idx_to_class.values():
    os.makedirs(os.path.join(OUTPUT_ROOT, class_name), exist_ok=True)

# === TÜM KLASÖRLERİ DOLAŞ ===
for board_folder in os.listdir(INPUT_ROOT):
    board_path = os.path.join(INPUT_ROOT, board_folder)

    if not os.path.isdir(board_path):
        continue

    print(f"[INFO] İşleniyor: {board_folder}")

    for img_name in os.listdir(board_path):
        img_path = os.path.join(board_path, img_name)

        # görüntü oku
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # resize + normalize
        img_resized = cv2.resize(img, IMG_SIZE)
        img_norm = img_resized / 255.0

        # shape: (1, 64, 64, 1)
        img_input = np.expand_dims(img_norm, axis=(0, -1))

        # tahmin
        preds = model.predict(img_input, verbose=0)
        class_idx = np.argmax(preds)
        class_name = idx_to_class[class_idx]

        # hedef klasör
        target_dir = os.path.join(OUTPUT_ROOT, class_name)

        # benzersiz isim (çakışma olmasın)
        new_name = f"{board_folder}_{img_name}"
        target_path = os.path.join(target_dir, new_name)

        # dosyayı kopyala (istersen move yapabilirsin)
        shutil.copy(img_path, target_path)

print("\n[BAŞARILI] Tüm taşlar sınıflara ayrıldı!")