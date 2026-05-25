import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- YOLLAR ---
NEW_DATASET_PATH = "sorted_pieces"
# Mevcut (eski) modelin yolu
PREVIOUS_MODEL_PATH = "models/chess_model_v3.keras"
# Yeni kaydedilecek model adı
FINE_TUNED_MODEL_PATH = "models/chess_model_finetuned.keras"
CLASS_INDICES_PATH = "models/class_indices.json"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
# Fine-tuning için genelde 15-20 epoch yeterli olur
EPOCHS = 25

# --- 1) VERİ YÜKLEME VE ÖZEL AUGMENTATION ---
# PDF'deki çizgili dokuyu ve bulanıklığı simüle etmek için
# biraz daha agresif augmentation ekliyoruz.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,      # Taşlar hafif yamuk olabilir
    width_shift_range=0.1,  # PDF kesim hataları için
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    NEW_DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    NEW_DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# --- 2) MEVCUT MODELİ YÜKLE ---
print(f"[INFO] Mevcut model yükleniyor: {PREVIOUS_MODEL_PATH}")
model = load_model(PREVIOUS_MODEL_PATH)

# --- 3) FINE-TUNING AYARLARI (KRİTİK KISIM) ---
# Mevcut bilgileri bozmamak için Learning Rate'i 10 kat düşürüyoruz (1e-4 veya 1e-5)
initial_learning_rate = 1e-4

model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- 4) CALLBACKS ---
# Kayıt ismini güncelledik
callbacks = [
    ModelCheckpoint(FINE_TUNED_MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1)
]

# --- 5) EĞİTİM ---
print("\n[INFO] Fine-tuning süreci başlıyor...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

print(f"\n[TAMAMLANDI] Yeni model kaydedildi: {FINE_TUNED_MODEL_PATH}")