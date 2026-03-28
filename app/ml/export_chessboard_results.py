import io
from pathlib import Path

from PIL import Image

from app.core.db import create_connection
from app.core.paths import ROOT_DIR


OUT_DIR = ROOT_DIR / "temp" / "chessboard_exports"
OUT_CHESS = OUT_DIR / "chessboard"
OUT_NOT = OUT_DIR / "not_chessboard"

LIMIT = None  #200 yazarsak ilk 200 kaydı export eder.


def main():
    OUT_CHESS.mkdir(parents=True, exist_ok=True)
    OUT_NOT.mkdir(parents=True, exist_ok=True)

    conn = create_connection()
    cur = conn.cursor()

    query = """
        SELECT f.image_id, f.is_chessboard, f.chessboard_score, p.blob
        FROM image_features f
        INNER JOIN pdf_images p ON p.id = f.image_id
        WHERE f.is_chessboard IS NOT NULL
    """
    if LIMIT is not None:
        query += f" LIMIT {int(LIMIT)}"

    cur.execute(query)
    rows = cur.fetchall()
    print(f"[INFO] Export edilecek kayıt sayısı: {len(rows)}")

    saved = 0
    failed = 0

    for row in rows:
        image_id, is_cb, score, blob = row

        try:
            img = Image.open(io.BytesIO(blob))
            img.load()
        except Exception as e:
            failed += 1
            print(f"[WARN] image_id={image_id} açılamadı: {e}")
            continue

        score_str = "NA" if score is None else f"{float(score):.4f}"
        filename = f"id{image_id}_score{score_str}.png"

        out_path = (OUT_CHESS if int(is_cb) == 1 else OUT_NOT) / filename

        try:
            img.save(out_path)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] image_id={image_id} kaydedilemedi: {e}")

    conn.close()
    print(f"[OK] Export tamamlandı | saved={saved} | failed={failed}")
    print(f"[DIR] {OUT_DIR}")


if __name__ == "__main__":
    main()