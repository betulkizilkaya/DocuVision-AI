import io
import sqlite3
from pathlib import Path

from PIL import Image

from app.core.paths import DB_PATH, ROOT_DIR


EXPORT_DIR = ROOT_DIR / "debug_exports" / "labeled_images"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def main(limit=None):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    base_query = """
        SELECT
            p.id AS image_id,
            p.blob AS blob,
            f.has_person,
            f.person_score,
            f.has_logo,
            f.logo_score,
            f.has_game_notation,
            f.game_notation_score,
            f.predicted_label,
            f.predicted_confidence
        FROM pdf_images p
        JOIN image_features f ON f.image_id = p.id
        ORDER BY p.id
    """

    if limit is None:
        rows = conn.execute(base_query).fetchall()
    else:
        rows = conn.execute(base_query + " LIMIT ?", (limit,)).fetchall()

    print(f"{len(rows)} görsel dışarı aktarılıyor...")

    for row in rows:
        image_id = row["image_id"]
        blob = row["blob"]

        try:
            img = Image.open(io.BytesIO(blob)).convert("RGB")

            label = row["predicted_label"] or "none"
            conf = row["predicted_confidence"] if row["predicted_confidence"] is not None else 0.0
            has_person = row["has_person"] if row["has_person"] is not None else 0
            has_logo = row["has_logo"] if row["has_logo"] is not None else 0
            has_game = row["has_game_notation"] if row["has_game_notation"] is not None else 0

            filename = (
                f"{image_id:06d}_"
                f"{label}_"
                f"conf-{conf:.2f}_"
                f"person-{has_person}_"
                f"logo-{has_logo}_"
                f"game-{has_game}.png"
            )

            img.save(EXPORT_DIR / filename)

        except Exception as e:
            print(f"[ERROR] image_id={image_id} kaydedilemedi: {e}")

    conn.close()
    print(f"Bitti. Klasör: {EXPORT_DIR}")


if __name__ == "__main__":
    main(limit=None)