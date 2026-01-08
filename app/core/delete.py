import sqlite3
from pathlib import Path

from app.core.paths import DB_PATH


def create_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(DB_PATH),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def reset_image_similarity() -> None:
    conn = create_connection()
    cur = conn.cursor()

    print("[INFO] Dropping image_similarity table if exists...")
    cur.execute("DROP TABLE IF EXISTS image_similarity;")

    print("[INFO] Creating image_similarity table...")
    cur.execute(
        """
        CREATE TABLE image_similarity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_id_a INTEGER NOT NULL,
            image_id_b INTEGER NOT NULL,

            ssim REAL,
            phash REAL,
            orb REAL,
            akaze REAL,

            label TEXT NOT NULL,
            decision_phase INTEGER NOT NULL,
            reason TEXT,

            FOREIGN KEY (image_id_a) REFERENCES pdf_images(id),
            FOREIGN KEY (image_id_b) REFERENCES pdf_images(id),

            UNIQUE (image_id_a, image_id_b)
        );
        """
    )

    conn.commit()
    conn.close()

    print("[✓] image_similarity table reset completed.")


if __name__ == "__main__":
    reset_image_similarity()
