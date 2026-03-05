#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from app.core.paths import DB_PATH, ROOT_DIR


IMG_SIZE = (64, 64)

CLASS_TO_FEN = {
    "White_Pawn": "P", "White_Rook": "R", "White_Knight": "N", "White_Bishop": "B",
    "White_Queen": "Q", "White_King": "K",
    "Black_Pawn": "p", "Black_Rook": "r", "Black_Knight": "n", "Black_Bishop": "b",
    "Black_Queen": "q", "Black_King": "k",
    "Empty_Square": None,
}


@dataclass
class FenParams:
    piece_model_path: str
    class_indices_path: str


def preprocess_cell(cell_bgr):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray, IMG_SIZE)
    x = res.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=(0, -1))


def load_piece_model_and_index(p: FenParams):
    model = load_model(p.piece_model_path)
    with open(p.class_indices_path, "r", encoding="utf-8") as f:
        idx_to_class = {int(v): k for k, v in json.load(f).items()}
    return model, idx_to_class


def board_img_to_fen(board_img, model, idx_to_class: Dict[int, str]) -> str:
    h_orig, w_orig = board_img.shape[:2]
    pad = int(w_orig * 0.05)
    clean = board_img[pad:h_orig - pad, pad:w_orig - pad]
    clean = cv2.resize(clean, (512, 512))

    cell_size = 512 // 8
    fen_rows = []

    for r in range(8):
        empty, fen_row = 0, ""
        for c in range(8):
            cell = clean[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size]
            x_input = preprocess_cell(cell)
            preds = model.predict(x_input, verbose=0)[0]
            cls_name = idx_to_class[int(np.argmax(preds))]
            char = CLASS_TO_FEN.get(cls_name)

            if char is None:
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += char
        if empty:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    return "/".join(fen_rows)


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok or buf is None:
        raise RuntimeError("PNG encode failed")
    return bytes(buf)


def ensure_tables(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chess_fen_multi (
            id INTEGER PRIMARY KEY,
            image_id INTEGER,
            board_index INTEGER,
            fen_format TEXT,
            UNIQUE(image_id, board_index)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS final_boards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            board_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            clf_score REAL,
            w INTEGER NOT NULL,
            h INTEGER NOT NULL,
            blob_png BLOB NOT NULL,
            created_at INTEGER NOT NULL,
            UNIQUE(image_id, board_index)
        )
        """
    )
    conn.commit()


def save_board_and_fen(
    conn: sqlite3.Connection,
    image_id: int,
    board_index: int,
    board_bgr: np.ndarray,
    source: str,
    clf_score: float,
    piece_model,
    idx_to_class: Dict[int, str],
) -> None:
    board_bgr = cv2.resize(board_bgr, (640, 640), interpolation=cv2.INTER_AREA)

    blob_png = encode_png(board_bgr)
    h, w = board_bgr.shape[:2]
    now = int(time.time())

    conn.execute(
        """
        INSERT OR REPLACE INTO final_boards
            (image_id, board_index, source, clf_score, w, h, blob_png, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (image_id, board_index, source, None if clf_score < 0 else float(clf_score), w, h, sqlite3.Binary(blob_png), now),
    )

    fen = board_img_to_fen(board_bgr, piece_model, idx_to_class)
    conn.execute(
        "INSERT OR REPLACE INTO chess_fen_multi (image_id, board_index, fen_format) VALUES (?, ?, ?)",
        (image_id, board_index, fen),
    )