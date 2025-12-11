# app/train_doc_classifier.py

import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "doc_type_dataset.csv"


def main():
    print(f"[OK] Dataset yükleniyor: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    train_doc_classifier(texts, labels, use_svm=True)


if __name__ == "__main__":
    main()
