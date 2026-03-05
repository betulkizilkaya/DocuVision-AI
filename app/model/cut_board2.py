#https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

INPUT_DIR = Path("data")
OUT_DIR = Path("cropped_by_corners")
DEBUG_DIR = Path("debug_corners_crop")

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

PATTERN_SIZE = (7, 7)  # 8x8 squares -> 7x7 inner corners


@dataclass
class Params:
    use_sb: bool = True
    classic_flags: int = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    # dışarıya kaç "kare" ekleyelim?
    # 1.0 => tam 1 kare (inner->outer sınır farkı)
    expand_cells: float = 1.00

    # güvenlik için biraz daha pay (piksel)
    extra_pad_px: int = 2

    # bazı görsellerde aşırı genişlemesin diye clamp
    max_expand_px: int = 300


def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def detect_corners_7x7(gray: np.ndarray, p: Params) -> Optional[np.ndarray]:
    """Return corners shape (49,2) if found else None."""
    if p.use_sb:
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE)
        if found and corners is not None:
            return corners.reshape(-1, 2).astype(np.float32)

    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, p.classic_flags)
    if not (found and corners is not None):
        return None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
    return corners2.reshape(-1, 2).astype(np.float32)


def estimate_cell_size_from_corners(corners49: np.ndarray) -> Tuple[float, float]:
    """
    corners49: (49,2) row-major => reshape (7,7,2)
    cell_w: medyan komşu x farkı
    cell_h: medyan komşu y farkı
    """
    C = corners49.reshape(7, 7, 2)
    xs = C[:, :, 0]
    ys = C[:, :, 1]

    # komşu farkları: x için aynı satırda sütun farkları
    dx = np.diff(xs, axis=1).reshape(-1)  # (7*6,)
    dy = np.diff(ys, axis=0).reshape(-1)  # (6*7,)

    # mutlak + medyan daha robust
    cell_w = float(np.median(np.abs(dx)))
    cell_h = float(np.median(np.abs(dy)))

    return cell_w, cell_h


def compute_outer_crop_box(corners49: np.ndarray, img_w: int, img_h: int, p: Params) -> Tuple[int, int, int, int]:
    """
    Returns (l, t, r, b) inclusive-ish crop box in image coords (we'll slice [t:b, l:r]).
    """
    cell_w, cell_h = estimate_cell_size_from_corners(corners49)

    # inner bounds
    min_x = float(np.min(corners49[:, 0]))
    max_x = float(np.max(corners49[:, 0]))
    min_y = float(np.min(corners49[:, 1]))
    max_y = float(np.max(corners49[:, 1]))

    # expand by 1 cell (or p.expand_cells)
    ex = min(cell_w * p.expand_cells, float(p.max_expand_px))
    ey = min(cell_h * p.expand_cells, float(p.max_expand_px))

    l = int(np.floor(min_x - ex - p.extra_pad_px))
    r = int(np.ceil (max_x + ex + p.extra_pad_px))
    t = int(np.floor(min_y - ey - p.extra_pad_px))
    b = int(np.ceil (max_y + ey + p.extra_pad_px))

    # clamp
    l = max(0, l)
    t = max(0, t)
    r = min(img_w - 1, r)
    b = min(img_h - 1, b)

    # güvenlik: çok dar/ters olmasın
    if r <= l or b <= t:
        return (0, 0, img_w - 1, img_h - 1)

    return (l, t, r, b)


def draw_debug(img_bgr: np.ndarray, corners49: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    vis = img_bgr.copy()
    l, t, r, b = box
    cv2.rectangle(vis, (l, t), (r, b), (0, 0, 255), 2)
    for (x, y) in corners49:
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
    return vis


def main():
    p = Params()

    for img_path in iter_images(INPUT_DIR):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = detect_corners_7x7(gray, p)

        if corners is None:
            print(f"NO corners: {img_path.name}")
            continue

        H, W = gray.shape[:2]
        l, t, r, b = compute_outer_crop_box(corners, W, H, p)

        cropped = img[t:b, l:r].copy()
        cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_crop.png"), cropped)

        dbg = draw_debug(img, corners, (l, t, r, b))
        cv2.imwrite(str(DEBUG_DIR / f"{img_path.stem}_debug.png"), dbg)

        cell_w, cell_h = estimate_cell_size_from_corners(corners)
        print(f"OK {img_path.name}  cell≈({cell_w:.1f}px, {cell_h:.1f}px)  crop=({l},{t})-({r},{b})")


if __name__ == "__main__":
    main()