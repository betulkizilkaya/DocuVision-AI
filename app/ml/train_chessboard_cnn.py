from pathlib import Path
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "data" / "chessboard_dataset"
MODEL_PATH = ROOT_DIR / "data" / "models" / "chessboard_cnn_v1.keras"

IMG_SIZE = (128, 128)
VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

SEED = 42
BATCH_SIZE = 32
EPOCHS = 15


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_images(folder: Path, label: int):
    X, y = [], []

    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue

        try:
            img = Image.open(p).convert("RGB").resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr)
            y.append(label)
        except Exception as e:
            print(f"[WARN] Atlandı: {p.name} -> {e}")

    return X, y


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    set_seed()

    chess_dir = DATASET_DIR / "chessboard"
    not_dir = DATASET_DIR / "not_chessboard"

    if not chess_dir.exists() or not not_dir.exists():
        raise SystemExit(
            "Dataset klasörleri bulunamadı: data/chessboard_dataset/chessboard ve not_chessboard"
        )

    X_pos, y_pos = load_images(chess_dir, 1)
    X_neg, y_neg = load_images(not_dir, 0)

    X = np.array(X_pos + X_neg, dtype=np.float32)
    y = np.array(y_pos + y_neg, dtype=np.int32)

    print(f"[INFO] Total samples: {len(y)} | Chess: {int(y.sum())} | Not: {int((y == 0).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    # Basit augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
    ])

    base_model = build_model()

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_augmentation(inputs)
    outputs = base_model(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, digits=3))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n[OK] Model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    main()