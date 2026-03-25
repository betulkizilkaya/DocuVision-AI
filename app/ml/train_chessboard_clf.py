from pathlib import Path
import numpy as np
import joblib
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


ROOT_DIR = Path(__file__).resolve().parents[2]  # ProjectNexus-Intelligent-PDF-Analysis/
DATASET_DIR = ROOT_DIR / "data" / "chessboard_dataset"
MODEL_PATH = ROOT_DIR / "data" / "models" / "chessboard_clf_v2.joblib"


IMG_SIZE = (64, 64)

def load_images(folder: Path, label: int):
    X, y = [], []
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in valid_exts:
            continue

        try:
            if not p.exists():
                print(f"[WARN] Dosya yok, atlandı: {p}")
                continue

            img = Image.open(p).convert("RGB").resize(IMG_SIZE)
            arr = np.array(img).astype(np.float32) / 255.0
            X.append(arr.flatten())
            y.append(label)

        except Exception as e:
            print(f"[WARN] Okunamadı: {p.name} -> {e}")

    return X, y

def main():
    chess_dir = DATASET_DIR / "chessboard"
    not_dir = DATASET_DIR / "not_chessboard"

    if not chess_dir.exists() or not not_dir.exists():
        raise SystemExit("Dataset klasörleri yok: data/chessboard_dataset/chessboard ve not_chessboard")

    X1, y1 = load_images(chess_dir, 1)
    X0, y0 = load_images(not_dir, 0)

    X = np.array(X1 + X0)
    y = np.array(y1 + y0)

    print("Total samples:", len(y), "Chess:", int(y.sum()), "Not:", int((y==0).sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # predict_proba garanti: LogisticRegression
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=3000))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, digits=3))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("\nSaved model:", MODEL_PATH)

if __name__ == "__main__":
    main()
