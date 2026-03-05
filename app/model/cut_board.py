#https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import csv
import sqlite3

import cv2
import numpy as np

from app.core.paths import DB_PATH


# ----------------------------
# Paths
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parent
DEBUG_DIR = ROOT_DIR / "debug_boards"  # Tahtaların kaydedileceği yer


# ----------------------------
# SQL
# ----------------------------
SQL = """
SELECT pi.id, pi.blob
FROM pdf_images pi
JOIN image_features f ON f.image_id = pi.id
WHERE f.is_chessboard = 1
  AND f.chessboard_score >= ?
"""


# ----------------------------
# Params
# ----------------------------
@dataclass
class Params:
    out_size: int = 640

    # Candidate block (board+caption) detection on whole page
    min_component_area_ratio: float = 0.004   # 0.4% of page
    max_component_area_ratio: float = 0.70
    min_wh: int = 110
    aspect_min: float = 0.50                  # caption increases height
    aspect_max: float = 2.20

    # Border detection inside each ROI
    canny1: int = 50
    canny2: int = 150
    hough_threshold: int = 60
    hough_min_len_ratio: float = 0.45         # more tolerant for broken scans
    hough_max_gap: int = 14
    line_angle_tol_deg: float = 12.0
    border_margin_ratio: float = 0.12         # border lines should be near ROI edges
    border_pad_ratio: float = 0.006           # crop a bit inside the frame

    # Fallback warp (when border not found)
    approx_eps_ratio: float = 0.02

    # Debug / limits
    save_intermediates: bool = True
    max_boards_per_image: int = 64
    max_candidates_per_page: int = 120


# ----------------------------
# Utilities
# ----------------------------

def _line_len(x1, y1, x2, y2) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def _cluster_1d(values: List[int], weights: List[float], tol: int) -> List[Tuple[int, float]]:
    """
    1D değerleri (x veya y) tolerans içinde kümeler.
    Her küme için: (merkez, toplam_ağırlık) döndürür.
    """
    if not values:
        return []
    idx = np.argsort(values)
    v = np.array(values, dtype=np.int32)[idx]
    w = np.array(weights, dtype=np.float32)[idx]

    clusters: List[Tuple[int, float]] = []
    cur_vals = [int(v[0])]
    cur_w = float(w[0])

    for i in range(1, len(v)):
        if abs(int(v[i]) - cur_vals[-1]) <= tol:
            cur_vals.append(int(v[i]))
            cur_w += float(w[i])
        else:
            center = int(np.median(cur_vals))
            clusters.append((center, cur_w))
            cur_vals = [int(v[i])]
            cur_w = float(w[i])

    center = int(np.median(cur_vals))
    clusters.append((center, cur_w))
    return clusters


def _intersect_hv(hline: Tuple[int, int, int, int], vline: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    hline: (x1,y1,x2,y2) yaklaşık yatay
    vline: (x1,y1,x2,y2) yaklaşık dikey
    Basit line intersection (sonsuz çizgiler).
    """
    x1, y1, x2, y2 = map(float, hline)
    x3, y3, x4, y4 = map(float, vline)

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return (np.nan, np.nan)

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
    return (px, py)


def warp_by_grid_lines(roi_bgr: np.ndarray, params: Params) -> Optional[np.ndarray]:
    """
    ROI içinde 9x9 grid çizgilerini yakalayıp dış sınırdan perspektif düzeltme yapar.
    Başarırsa out_size x out_size döndürür, yoksa None.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, params.canny1, params.canny2)

    h, w = edges.shape[:2]
    min_len = int(0.35 * min(h, w))          # grid çizgileri için daha düşük eşik
    max_gap = max(10, params.hough_max_gap)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(40, params.hough_threshold - 20),
        minLineLength=min_len,
        maxLineGap=max_gap
    )
    if lines is None:
        return None

    tol = params.line_angle_tol_deg

    # yatay/dikey adayları topla
    hy_vals: List[int] = []
    hy_w: List[float] = []
    hx_lines: List[Tuple[int, int, int, int]] = []

    vx_vals: List[int] = []
    vx_w: List[float] = []
    vy_lines: List[Tuple[int, int, int, int]] = []

    for x1, y1, x2, y2 in lines[:, 0]:
        ang = _angle_deg(x1, y1, x2, y2)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        L = _line_len(x1, y1, x2, y2)

        # yatay
        if dy < dx and (ang <= tol or ang >= 180 - tol):
            y = int((y1 + y2) / 2)
            hy_vals.append(y)
            hy_w.append(L)
            hx_lines.append((x1, y1, x2, y2))

        # dikey
        if dx < dy and (abs(ang - 90) <= tol):
            x = int((x1 + x2) / 2)
            vx_vals.append(x)
            vx_w.append(L)
            vy_lines.append((x1, y1, x2, y2))

    if len(hy_vals) < 6 or len(vx_vals) < 6:
        return None

    # kümeler: tol piksel (çizgi kalınlığı + tarama jitter)
    tol_px = max(6, int(0.012 * min(h, w)))  # ~%1.2
    hy_clusters = _cluster_1d(hy_vals, hy_w, tol=tol_px)
    vx_clusters = _cluster_1d(vx_vals, vx_w, tol=tol_px)

    # grid’de ideal 9 çizgi kümesi var. En güçlü 9 kümeyi seç.
    hy_clusters.sort(key=lambda t: t[1], reverse=True)
    vx_clusters.sort(key=lambda t: t[1], reverse=True)

    hy_sel = sorted([c[0] for c in hy_clusters[:9]])
    vx_sel = sorted([c[0] for c in vx_clusters[:9]])

    # En az 6 çizgi kümesi yoksa güvenme
    if len(hy_sel) < 6 or len(vx_sel) < 6:
        return None

    # Dış sınırlar: min/max (grid dış çizgileri)
    top_y = int(min(hy_sel))
    bot_y = int(max(hy_sel))
    left_x = int(min(vx_sel))
    right_x = int(max(vx_sel))

    # Bu x/y değerlerine en yakın gerçek hough segmentlerini seçelim
    def _pick_nearest_h(y_target: int) -> Optional[Tuple[int,int,int,int]]:
        best = None
        bestd = 1e18
        for (x1,y1,x2,y2) in hx_lines:
            y = int((y1+y2)/2)
            d = abs(y - y_target)
            if d < bestd:
                bestd = d
                best = (x1,y1,x2,y2)
        return best if bestd <= tol_px*3 else None

    def _pick_nearest_v(x_target: int) -> Optional[Tuple[int,int,int,int]]:
        best = None
        bestd = 1e18
        for (x1,y1,x2,y2) in vy_lines:
            x = int((x1+x2)/2)
            d = abs(x - x_target)
            if d < bestd:
                bestd = d
                best = (x1,y1,x2,y2)
        return best if bestd <= tol_px*3 else None

    h_top = _pick_nearest_h(top_y)
    h_bot = _pick_nearest_h(bot_y)
    v_left = _pick_nearest_v(left_x)
    v_right = _pick_nearest_v(right_x)

    if h_top is None or h_bot is None or v_left is None or v_right is None:
        return None

    # köşeler (kesişim)
    tl = _intersect_hv(h_top, v_left)
    tr = _intersect_hv(h_top, v_right)
    br = _intersect_hv(h_bot, v_right)
    bl = _intersect_hv(h_bot, v_left)

    pts = np.array([tl, tr, br, bl], dtype=np.float32)
    if np.any(np.isnan(pts)):
        return None

    # ROI içinde mi?
    if not (0 <= pts[:,0].min() < w and 0 <= pts[:,0].max() < w and 0 <= pts[:,1].min() < h and 0 <= pts[:,1].max() < h):
        return None

    # koordinat yazıları çizgi dışındaysa zaten dışarıda kalır.
    # yine de çok az içeri pad (sadece 1-2 px)
    pad = max(1, int(0.003 * min(w, h)))
    pts[0] += (pad, pad)          # TL
    pts[1] += (-pad, pad)         # TR
    pts[2] += (-pad, -pad)        # BR
    pts[3] += (pad, -pad)         # BL

    pts = order_points(pts)

    dst = np.array(
        [[0, 0], [params.out_size - 1, 0], [params.out_size - 1, params.out_size - 1], [0, params.out_size - 1]],
        dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(roi_bgr, M, (params.out_size, params.out_size))
    return warped

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def decode_blob(blob: bytes) -> Optional[np.ndarray]:
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def find_contours(binary_img: np.ndarray):
    """
    OpenCV 3/4 uyumlu findContours wrapper.
    Her zaman (list(contours), hierarchy) döndürür.
    """
    res = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 3:
        _, contours, hierarchy = res
    else:
        contours, hierarchy = res
    return list(contours), hierarchy


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
    return rect


def preprocess_binary(img_bgr: np.ndarray) -> np.ndarray:
    """
    Whole-page binarization to connect board + caption into components.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    # close to merge broken strokes (caption + frame + grid)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    return thr


def find_candidate_boxes(img_bgr: np.ndarray, params: Params) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Returns candidate bounding boxes (x,y,w,h) likely containing board (+ caption).
    Also returns thr for debug.
    """
    thr = preprocess_binary(img_bgr)
    H, W = thr.shape[:2]
    area_img = H * W

    num, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)

    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if w < params.min_wh or h < params.min_wh:
            continue
        if a < area_img * params.min_component_area_ratio:
            continue
        if a > area_img * params.max_component_area_ratio:
            continue

        aspect = w / float(h) if h else 0.0
        if not (params.aspect_min <= aspect <= params.aspect_max):
            continue

        boxes.append((x, y, w, h))

    # big to small
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    boxes = boxes[: params.max_candidates_per_page]

    return boxes, thr


def _angle_deg(x1, y1, x2, y2) -> float:
    ang = float(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
    ang = (ang + 180.0) % 180.0
    return ang


def find_border_rect_in_roi(roi_bgr: np.ndarray, params: Params) -> Optional[Tuple[int, int, int, int]]:
    """
    Find axis-aligned outer frame inside roi using long Hough lines near edges.
    Returns (l, t, r, b) in ROI coords.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, params.canny1, params.canny2)

    h, w = edges.shape[:2]
    min_len = int(params.hough_min_len_ratio * min(h, w))

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=params.hough_threshold,
        minLineLength=min_len,
        maxLineGap=params.hough_max_gap
    )
    if lines is None:
        return None

    tol = params.line_angle_tol_deg
    margin_x = int(params.border_margin_ratio * w)
    margin_y = int(params.border_margin_ratio * h)

    horizontals: List[int] = []
    verticals: List[int] = []

    for x1, y1, x2, y2 in lines[:, 0]:
        ang = _angle_deg(x1, y1, x2, y2)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # horizontal
        if dy < dx and (ang <= tol or ang >= 180 - tol):
            y = int((y1 + y2) / 2)
            if y < margin_y or y > (h - margin_y):
                horizontals.append(y)

        # vertical
        if dx < dy and (abs(ang - 90) <= tol):
            x = int((x1 + x2) / 2)
            if x < margin_x or x > (w - margin_x):
                verticals.append(x)

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    top = int(np.min(horizontals))
    bottom = int(np.max(horizontals))
    left = int(np.min(verticals))
    right = int(np.max(verticals))

    if right <= left or bottom <= top:
        return None

    # must be reasonably large relative to ROI
    if (right - left) < int(0.45 * w) or (bottom - top) < int(0.45 * h):
        return None

    # crop slightly inside the frame
    pad = max(2, int(params.border_pad_ratio * min(w, h)))
    left = max(0, left + pad)
    right = min(w - 1, right - pad)
    top = max(0, top + pad)
    bottom = min(h - 1, bottom - pad)

    ww = right - left
    hh = bottom - top
    if ww <= 0 or hh <= 0:
        return None

    # after removing caption, should be near-square
    aspect = ww / float(hh)
    if not (0.80 <= aspect <= 1.28):
        return None

    return (left, top, right, bottom)


def fallback_warp_biggest_quad(roi_bgr: np.ndarray, params: Params) -> Optional[np.ndarray]:
    """
    If border not found: find biggest contour inside ROI and warp it.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = find_contours(thr)
    if len(contours) == 0:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, params.approx_eps_ratio * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(cnt)
        quad = cv2.boxPoints(rect).astype(np.float32)

    quad = order_points(quad)

    dst = np.array(
        [[0, 0], [params.out_size - 1, 0], [params.out_size - 1, params.out_size - 1], [0, params.out_size - 1]],
        dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(roi_bgr, M, (params.out_size, params.out_size))
    return warped


def crop_and_normalize(board_bgr: np.ndarray, out_size: int) -> np.ndarray:
    return cv2.resize(board_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)


def extract_boards_from_image(
    img_bgr: np.ndarray,
    params: Params
) -> Tuple[List[np.ndarray], np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Extract boards from a whole page image.
    Returns: boards, thr(page), candidate boxes
    """
    cand_boxes, thr = find_candidate_boxes(img_bgr, params)

    boards: List[np.ndarray] = []

    for (x, y, w, h) in cand_boxes:
        roi = img_bgr[y:y + h, x:x + w].copy()

        # 1) önce grid'e göre warp dene (en sağlam)
        warped = warp_by_grid_lines(roi, params)

        if warped is not None:
            boards.append(warped)
        else:
            # 2) grid de olmazsa, eski border/crop yolun (çoğu düzgün çıkarıyordu)
            border = find_border_rect_in_roi(roi, params)
            if border is not None:
                l, t, r, b = border
                board = roi[t:b, l:r].copy()
                board = crop_and_normalize(board, params.out_size)
                boards.append(board)
            else:
                # 3) en son çare: eski fallback
                fb = fallback_warp_biggest_quad(roi, params)
                if fb is not None:
                    fb = crop_and_normalize(fb, params.out_size)
                    boards.append(fb)

        if len(boards) >= params.max_boards_per_image:
            break

    return boards, thr, cand_boxes


def draw_boxes(img_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis, f"cand_{i}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2, cv2.LINE_AA
        )
    return vis


def save_debug(
    image_id: int,
    img_bgr: np.ndarray,
    thr: np.ndarray,
    cand_boxes: List[Tuple[int, int, int, int]],
    boards: List[np.ndarray],
    params: Params
) -> None:
    out_dir = DEBUG_DIR / str(image_id)
    ensure_dir(out_dir)

    if params.save_intermediates:
        cv2.imwrite(str(out_dir / "orig.png"), img_bgr)
        cv2.imwrite(str(out_dir / "thr.png"), thr)
        cv2.imwrite(str(out_dir / "candidates.png"), draw_boxes(img_bgr, cand_boxes))

    for i, b in enumerate(boards, start=1):
        cv2.imwrite(str(out_dir / f"board_{i:02d}.png"), b)


def run(score_threshold: float = 0.8) -> None:
    ensure_dir(DEBUG_DIR)
    index_csv = DEBUG_DIR / "index.csv"

    params = Params()

    with sqlite3.connect(str(DB_PATH)) as conn, open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "num_boards", "note"])

        cur = conn.cursor()
        cur.execute(SQL, (score_threshold,))
        rows = cur.fetchall()

        for image_id, blob in rows:
            img = decode_blob(blob)
            if img is None:
                writer.writerow([image_id, 0, "decode_failed"])
                continue

            boards, thr, cand_boxes = extract_boards_from_image(img, params)
            save_debug(image_id, img, thr, cand_boxes, boards, params)

            note = "ok" if boards else "no_board_found"
            writer.writerow([image_id, len(boards), note])

    print("Bitti. Çıktılar:", DEBUG_DIR)


if __name__ == "__main__":
    run(score_threshold=0.8)