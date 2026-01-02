import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- AYARLAR ---
DATASET_PATH = "train_250"
BASE_MODEL_PATH = "models/chess_model_v3.keras"
FINE_TUNED_MODEL_PATH = "models/chess_model_v3_finetuned.keras"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 3          # <-- AZ
LEARNING_RATE = 1e-4  # <-- KÜÇÜK

# --- VERİ ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# --- MODELİ YÜKLE ---
model = load_model(BASE_MODEL_PATH)

# --- FINE-TUNE AYARI ---
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- EĞİT ---
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[
        EarlyStopping(patience=1, restore_best_weights=True)
    ]
)

# --- KAYDET ---
model.save(FINE_TUNED_MODEL_PATH)
print(f"[OK] Fine-tune tamamlandı → {FINE_TUNED_MODEL_PATH}")
