#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import numpy as np
import cv2
import joblib

from app.core.paths import DB_PATH, ROOT_DIR

from stage_crop import (
    RoiParams, CornerParams, HoughParams, ClfParams,
    extract_final_boards_from_page,
)

from stage_fen import (
    FenParams,
    ensure_tables,
    load_piece_model_and_index,
    save_board_and_fen,
)

MIN_ID = 150
MAX_ID = 300
MIN_CHESSBOARD_SCORE = 0.35

SQL_PAGES = """
SELECT pi.id, pi.blob
FROM pdf_images pi
JOIN image_features f ON f.image_id = pi.id
WHERE f.is_chessboard = 1
  AND f.chessboard_score >= ?
  AND pi.id BETWEEN ? AND ?
"""

# models
PIECE_MODEL_PATH = ROOT_DIR / "app" / "model" / "models" / "chess_model_v3.keras"
CLASS_INDICES_PATH = ROOT_DIR / "app" / "model" / "models" / "class_indices.json"
CHESSBOARD_CLF_PATH = ROOT_DIR / "data" / "models" / "chessboard_clf.joblib"


def decode_page_blob(blob: bytes):
    arr = np.frombuffer(blob, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    # load models once
    clf = joblib.load(str(CHESSBOARD_CLF_PATH))

    fen_p = FenParams(piece_model_path=str(PIECE_MODEL_PATH), class_indices_path=str(CLASS_INDICES_PATH))
    piece_model, idx_to_class = load_piece_model_and_index(fen_p)

    # params
    roi_p = RoiParams()
    corner_p = CornerParams()
    hough_p = HoughParams(out_size=640)
    clf_p = ClfParams(img_size=(64, 64), proba_threshold=0.50)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_tables(conn)

    rows = conn.execute(
        SQL_PAGES,
        (MIN_CHESSBOARD_SCORE, MIN_ID, MAX_ID)
    ).fetchall()
    print(f"[RUN] {len(rows)} sayfa taranıyor...")

    for row in rows:
        image_id = int(row["id"])
        page = decode_page_blob(row["blob"])
        if page is None:
            continue

        finals = extract_final_boards_from_page(
            page_bgr=page,
            clf=clf,
            roi_p=roi_p,
            corner_p=corner_p,
            hough_p=hough_p,
            clf_p=clf_p,
        )

        if not finals:
            print(f"[RUN] image_id={image_id}: final yok")
            continue

        for i, (board_bgr, source, score) in enumerate(finals):
            try:
                save_board_and_fen(
                    conn=conn,
                    image_id=image_id,
                    board_index=i,
                    board_bgr=board_bgr,
                    source=source,
                    clf_score=score,
                    piece_model=piece_model,
                    idx_to_class=idx_to_class,
                )
                print(f"[OK] image_id={image_id} board={i} source={source} score={score:.3f}")
            except Exception as e:
                print(f"[ERR] image_id={image_id} board={i}: {e}")

        conn.commit()

    conn.close()
    print("[RUN] bitti.")


if __name__ == "__main__":
    main()