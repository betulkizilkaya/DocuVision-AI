# app/ml/train_doc_classifier.py

from pathlib import Path
import pandas as pd

# Model trainer fonksiyonun nerede ise ona göre import:
# Senin paylaştığın yapıda bu fonksiyon app/text/document_classifier.py içinde.
from app.text.document_classifier import train_doc_classifier

# ------------------------------------------------------------
# Proje kökü: .../ProjectNexus-Intelligent-PDF-Analysis
# Bu dosya: app/ml/train_doc_classifier.py => parents[2] proje kökü
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset: proje_kökü/data/doc_type_dataset.csv
DATASET_PATH = PROJECT_ROOT / "data" / "doc_type_dataset.csv"

# Model: proje_kökü/data/models/doc_type_clf.joblib
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "doc_type_clf.joblib"


def main() -> None:
    print(f"[OK] Dataset yükleniyor: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {DATASET_PATH}\n"
            f"Önce şunu üret: python -m app.ml.build_doc_type_dataset"
        )

    df = pd.read_csv(DATASET_PATH)

    # Beklenen kolonlar: text, label
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV kolonları hatalı. Beklenen: text,label | Bulunan: {list(df.columns)}")

    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].fillna("").astype(str).tolist()

    # Boş kayıtları ayıkla
    cleaned = [(t, y) for (t, y) in zip(texts, labels) if t.strip() and y.strip()]
    if not cleaned:
        raise RuntimeError("Dataset boş/etiketsiz görünüyor (text/label).")

    texts, labels = zip(*cleaned)

    print(f"[OK] Eğitim başlıyor. Örnek sayısı: {len(texts)} | Sınıf sayısı: {len(set(labels))}")
    train_doc_classifier(list(texts), list(labels), model_path=MODEL_PATH, use_svm=True)

    print(f"[OK] Model yazıldı: {MODEL_PATH}")


if __name__ == "__main__":
    main()
