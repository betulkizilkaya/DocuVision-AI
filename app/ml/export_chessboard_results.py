import sqlite3, io
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]  # repo root
DB_PATH = ROOT_DIR / "db" / "corpus.sqlite"

OUT_DIR = ROOT_DIR / "temp" / "chessboard_exports"
OUT_CHESS = OUT_DIR / "chessboard"
OUT_NOT = OUT_DIR / "not_chessboard"


LIMIT = None  # hepsi için None, denemek için 200 gibi sayı yazabilirsin

def main():
    OUT_CHESS.mkdir(parents=True, exist_ok=True)
    OUT_NOT.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()

    # Sadece sınıflandırılmış olanları al (NULL olmayanlar)
    q = """
    SELECT image_id, is_chessboard, chessboard_score
    FROM image_features
    WHERE is_chessboard IS NOT NULL
    """
    if LIMIT:
        q += f" LIMIT {int(LIMIT)}"

    cur.execute(q)
    rows = cur.fetchall()
    print(f"[INFO] Export edilecek kayıt: {len(rows)}")

    for image_id, is_cb, score in rows:
        cur.execute("SELECT blob FROM pdf_images WHERE id=?", (image_id,))
        r = cur.fetchone()
        if not r:
            continue
        blob = r[0]

        try:
            img = Image.open(io.BytesIO(blob))
            img.load()
        except Exception as e:
            print(f"[WARN] image_id={image_id} açılamadı: {e}")
            continue

        score_str = "NA" if score is None else f"{float(score):.2f}"
        fname = f"id{image_id}_score{score_str}.png"

        out_path = (OUT_CHESS if int(is_cb) == 1 else OUT_NOT) / fname
        try:
            img.save(out_path)
        except Exception as e:
            print(f"[WARN] image_id={image_id} kaydedilemedi: {e}")

    con.close()
    print(f"[OK] Bitti. Klasör: {OUT_DIR}")

if __name__ == "__main__":
    main()
