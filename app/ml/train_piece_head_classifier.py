from __future__ import annotations

from pathlib import Path
from collections import Counter

import joblib
import numpy as np
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT_DIR / "temp" / "piece_head_dataset_clean"
MODEL_PATH = ROOT_DIR / "data" / "models" / "piece_head_clf.joblib"

CLASSES = ["B", "N", "R", "Q", "K", "P", "empty_or_noise"]

IMG_SIZE = (32, 32)


def load_images(folder: Path, label: str) -> tuple[list[np.ndarray], list[str]]:
    X = []
    y = []

    for p in folder.iterdir():
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            continue

        img = Image.open(p).convert("L").resize(IMG_SIZE)
        arr = np.array(img).astype(np.float32) / 255.0

        X.append(arr.flatten())
        y.append(label)

    return X, y


def main() -> None:
    X_all = []
    y_all = []

    for cls in CLASSES:
        folder = DATASET_DIR / cls

        if not folder.exists():
            print(f"[WARN] Klasör yok: {folder}")
            continue

        X, y = load_images(folder, cls)
        X_all.extend(X)
        y_all.extend(y)

    if not X_all:
        raise SystemExit("Hiç veri bulunamadı. Önce create_dataset.py çalıştır.")

    X = np.array(X_all)
    y = np.array(y_all)

    print(f"[INFO] Toplam örnek: {len(y)}")
    counts = Counter(y)
    for cls in CLASSES:
        print(f"  {cls}: {counts.get(cls, 0)}")

    if len(set(y)) < 2:
        raise SystemExit("En az 2 sınıf gerekli.")

    min_count = min(counts.values())

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs"
        )),
    ])

    if min_count < 2:
        print("[WARN] Bazı sınıflarda çok az örnek var. Tüm veriyle eğitim yapılacak.")
        model.fit(X, y)

    else:
        test_size = 0.2

        # test setinde her sınıftan en az 1 örnek olsun diye kontrol
        if len(y) * test_size < len(set(y)):
            test_size = len(set(y)) / len(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=CLASSES))

        print("\nClassification report:")
        print(classification_report(
            y_test,
            y_pred,
            labels=CLASSES,
            zero_division=0,
            digits=3,
        ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n[OK] Model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    main()