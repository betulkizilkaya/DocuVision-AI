import sys
from pathlib import Path

import pdfplumber
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.paths import DATA_DIR

RAW_DATASET_DIR = DATA_DIR / "doc_type_raw"
OUTPUT_CSV = DATA_DIR / "doc_type_dataset.csv"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 300
MIN_DOCUMENT_TEXT_LEN = 300

EMPTY_LABEL = "image_only_or_empty"

TOURNAMENT_HINTS = [
    "tournament",
    "chess festival",
    "rapid tournament",
    "blitz tournament",
    "open chess",
    "championship",
    "standings",
    "ranking",
    "rank",
    "pts",
    "tb1",
    "tb2",
    "federation",
    "fide",
    "arbiter",
    "pairing",
    "round",
    "results",
    "bulletin",
]


def clean_text(text: str) -> str:
    return " ".join(text.split())


def looks_like_tournament(text: str) -> bool:
    text = clean_text(text).lower()
    return any(keyword in text for keyword in TOURNAMENT_HINTS)


def chunk_text(text: str):
    text = clean_text(text)
    chunks = []

    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

        if start >= len(text):
            break

    return chunks


def extract_pdf_chunks(pdf_path: Path, label: str):
    rows = []
    full_text_parts = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = clean_text(page.extract_text() or "")
            full_text_parts.append(text)

            for chunk_id, chunk in enumerate(chunk_text(text)):
                rows.append({
                    "filename": pdf_path.name,
                    "file_path": str(pdf_path),
                    "page_no": page_no,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "label": label,
                    "source_type": "pdf_text",
                })

    full_text = clean_text(" ".join(full_text_parts))

    if len(full_text) < MIN_DOCUMENT_TEXT_LEN:
        if looks_like_tournament(full_text):
            return [{
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "page_no": 0,
                "chunk_id": 0,
                "text": full_text if full_text else "__TOURNAMENT_LOW_TEXT__",
                "label": "tournament_report",
                "source_type": "low_text_tournament",
            }]

        return [{
            "filename": pdf_path.name,
            "file_path": str(pdf_path),
            "page_no": 0,
            "chunk_id": 0,
            "text": "__NO_EXTRACTABLE_TEXT__",
            "label": EMPTY_LABEL,
            "source_type": "empty_or_scanned",
        }]

    return rows


def main():
    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset klasörü bulunamadı: {RAW_DATASET_DIR}\n"
            f"Şu yapıda oluştur:\n"
            f"data/doc_type_raw/educational_chess/*.pdf\n"
            f"data/doc_type_raw/tournament_report/*.pdf\n"
            f"data/doc_type_raw/image_only_or_empty/*.pdf"
        )

    rows = []

    label_dirs = [p for p in RAW_DATASET_DIR.iterdir() if p.is_dir()]

    if not label_dirs:
        raise RuntimeError(f"{RAW_DATASET_DIR} içinde sınıf klasörü yok.")

    for label_dir in label_dirs:
        label = label_dir.name
        pdf_files = sorted(label_dir.glob("*.pdf"))

        print(f"\n[LABEL] {label} | PDF sayısı: {len(pdf_files)}")

        for pdf_path in pdf_files:
            print(f"  [OK] İşleniyor: {pdf_path.name}")

            try:
                pdf_rows = extract_pdf_chunks(pdf_path, label)
            except Exception as e:
                print(f"  [HATA] Okunamadı: {pdf_path.name} | {e}")
                continue

            rows.extend(pdf_rows)

    if not rows:
        raise RuntimeError("Hiç veri üretilemedi.")

    df = pd.DataFrame(rows)
    df = df[df["text"].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=["filename", "page_no", "chunk_id", "text", "label"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\n[OK] Dataset oluşturuldu.")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Toplam chunk sayısı: {len(df)}")

    print("\nSınıf dağılımı:")
    print(df["label"].value_counts())

    print("\nPDF dağılımı:")
    print(df.groupby("label")["filename"].nunique())

    print("\nKaynak tipi dağılımı:")
    print(df["source_type"].value_counts())


if __name__ == "__main__":
    main()