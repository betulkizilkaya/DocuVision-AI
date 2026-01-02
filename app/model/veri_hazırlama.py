import os
import shutil
import random

# --- AYARLAR ---
BIG_DATASET_PATH = "train"   # büyük taş datasetin
OUTPUT_DATASET_PATH = "train_250"  # oluşturulacak yeni eğitim seti
SAMPLES_PER_CLASS = 250

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

random.seed(42)  # tekrarlanabilirlik için

# --- KONTROLLER ---
if not os.path.isdir(BIG_DATASET_PATH):
    raise FileNotFoundError(f"Büyük dataset bulunamadı: {BIG_DATASET_PATH}")

os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)

# --- ANA İŞ ---
for class_name in os.listdir(BIG_DATASET_PATH):
    class_src = os.path.join(BIG_DATASET_PATH, class_name)
    if not os.path.isdir(class_src):
        continue

    images = [
        f for f in os.listdir(class_src)
        if f.lower().endswith(VALID_EXT)
    ]

    if len(images) < SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"{class_name} sınıfında yeterli görsel yok "
            f"({len(images)} / {SAMPLES_PER_CLASS})"
        )

    selected = random.sample(images, SAMPLES_PER_CLASS)

    class_dst = os.path.join(OUTPUT_DATASET_PATH, class_name)
    os.makedirs(class_dst, exist_ok=True)

    for fname in selected:
        src = os.path.join(class_src, fname)
        dst = os.path.join(class_dst, fname)
        shutil.copy2(src, dst)

    print(f"[OK] {class_name}: {SAMPLES_PER_CLASS} görsel kopyalandı")

print("\n[BAŞARILI] train_250 dataseti hazır.")



