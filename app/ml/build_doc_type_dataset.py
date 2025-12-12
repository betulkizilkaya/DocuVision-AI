# app/build_doc_type_dataset.py

import pdfplumber
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Her PDF'e elle bir label veriyoruz
    pdf_labels = {
        "texas_pdf.pdf": "tournament_report",
        "Texas-Knights - Nov-Dec-2023.pdf": "tournament_report",
        "Encyclopedia of Chess Miniatures (2014).pdf": "book_chapter",
        "Test Your Chess - Assess and Improve Your Chess Skills.pdf": "book_chapter",
        "OSD Satranç Ders Notları.pdf": "book_chapter",
        "satrancailkadim.pdf": "book_chapter",
    }

    rows = []

    for filename, label in pdf_labels.items():
        pdf_path = DATA_DIR / filename
        if not pdf_path.exists():
            print(f"[Uyarı] PDF bulunamadı, atlanıyor: {pdf_path}")
            continue

        print(f"[OK] PDF okunuyor: {pdf_path.name}")
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        rows.append({"text": text, "label": label})

    if not rows:
        print("Hiç veri yok, CSV oluşturulmadı.")
        return

    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "doc_type_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Dataset kaydedildi: {csv_path}")
    print(df.head())


if __name__ == "__main__":
    main()
