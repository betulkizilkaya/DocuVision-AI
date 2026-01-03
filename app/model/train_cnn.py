import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input,
    BatchNormalization, Activation
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

DATASET_PATH = "train_250"

MODEL_SAVE_PATH = "models/chess_model_v3.keras"   # .keras önerilir (TF2+)
CLASS_INDICES_PATH = "models/class_indices.json"

IMG_SIZE = (64, 64)     # 32 yerine 64: taş detayları için daha güvenli
BATCH_SIZE = 32
EPOCHS = 50             # EarlyStopping ile zaten gerektiği kadar sürecek

if not os.path.exists("models"):
    os.makedirs("models")

# --- 1) VERİ YÜKLEME / AUGMENTATION ---
# Eğitim için augmentation: genelleme için şart
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    brightness_range=(0.8, 1.2)
)

# Validasyon için SADECE rescale: ölçüm setini "kirletmemek" için
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print(f"[INFO] '{DATASET_PATH}' klasöründen veriler okunuyor...")

try:
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False  # metriklerin tutarlı olması için
    )
except Exception:
    print(f"HATA: Veri okunamadı. Klasör yolunu kontrol et: {DATASET_PATH}")
    raise

# Sınıf isimlerini kaydet
class_indices = train_generator.class_indices
with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_indices, f, ensure_ascii=False, indent=2)

num_classes = len(class_indices)
print(f"[INFO] Tespit edilen sınıflar ({num_classes} adet): {list(class_indices.keys())}")

# --- 2) CLASS IMBALANCE KONTROLÜ (200'er adetse ağırlık vermeye gerek yok) ---
# Yine de doğrulama: dizin hatası / eksik klasör durumunda sessizce bozulmasın
counts = np.bincount(train_generator.classes)
is_balanced = (counts.max() == counts.min())

class_weight = None
if not is_balanced:
    # dengesizse otomatik ağırlık
    total = counts.sum()
    class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    print("[WARN] Sınıf dağılımı dengesiz görünüyor. class_weight uygulanacak:", class_weight)
else:
    print("[INFO] Sınıf dağılımı dengeli görünüyor (muhtemelen her sınıf ~200). class_weight yok.")

# --- 3) MODEL ---
# BatchNorm + daha stabil blok yapısı
model = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

    Conv2D(32, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), padding="same"),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])

# LR’i açıkça belirlemek daha kontrollü
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- 4) CALLBACKS: erken durdurma + en iyi modeli kaydet + LR düşür ---
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

# --- 5) EĞİTİM ---
print("\n[INFO] Eğitim başlıyor...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weight
)

print(f"\n[BAŞARILI] En iyi model kaydedildi: {MODEL_SAVE_PATH}")
print(f"[INFO] class_indices kaydedildi: {CLASS_INDICES_PATH}")

# --- 6) GRAFİK ---
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Satranç Taşı Tanıma Başarısı")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("models/basari_grafigi.png", dpi=150)
print("[INFO] Grafik 'models/basari_grafigi.png' dosyasına kaydedildi.")
