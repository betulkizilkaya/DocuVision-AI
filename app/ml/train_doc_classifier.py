from pathlib import Path

import pandas as pd
import pdfplumber

from app.text.document_classifier import train_doc_classifier, load_doc_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "doc_type_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "doc_type_clf.joblib"

DATA_DIR = PROJECT_ROOT / "data"

MIN_DOCUMENT_TEXT_LEN = 300


def clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_full_text(pdf_path: Path) -> str:
    parts = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")

    return clean_text(" ".join(parts))


def predict_external_pdfs():
    model = load_doc_classifier(MODEL_PATH)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("\n[INFO] data/ klasöründe test için PDF bulunamadı.")
        return

    print("\n[BAĞIMSIZ TEST / DATA KLASÖRÜ PDF TAHMİNLERİ]")

    for pdf_path in pdf_files:
        try:
            text = extract_full_text(pdf_path)
        except Exception as e:
            print(f"[HATA] {pdf_path.name} okunamadı: {e}")
            continue

        if len(text.strip()) < MIN_DOCUMENT_TEXT_LEN:
            pred = "image_only_or_empty"
        else:
            pred = model.predict([text])[0]

        print(f"{pdf_path.name} -> {pred}")


def main() -> None:
    print(f"[OK] Dataset yükleniyor: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {DATASET_PATH}\n"
            f"Önce çalıştır:\n"
            f"python -m app.ml.build_doc_type_dataset"
        )

    df = pd.read_csv(DATASET_PATH)

    required_cols = {"text", "label", "filename"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV kolonları eksik: {missing}\n"
            f"Bulunan kolonlar: {list(df.columns)}"
        )

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str)
    df["filename"] = df["filename"].fillna("").astype(str)

    df = df[df["text"].str.strip().astype(bool)]
    df = df[df["label"].str.strip().astype(bool)]
    df = df[df["filename"].str.strip().astype(bool)]

    if df.empty:
        raise RuntimeError("Dataset boş görünüyor.")

    texts = df["text"].tolist()
    labels = df["label"].tolist()
    groups = df["filename"].tolist()

    print(f"[OK] Eğitim başlıyor.")
    print(f"Toplam chunk: {len(texts)}")
    print(f"Sınıflar: {sorted(set(labels))}")

    print("\nSınıf dağılımı:")
    print(df["label"].value_counts())

    print("\nPDF dağılımı:")
    print(df.groupby("label")["filename"].nunique())

    train_doc_classifier(
        texts=texts,
        labels=labels,
        groups=groups,
        model_path=MODEL_PATH,
        use_svm=True,
        test_size=0.2,
    )

    print(f"\n[OK] Model yazıldı: {MODEL_PATH}")

    predict_external_pdfs()


if __name__ == "__main__":
    main()