#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# -----------------------------
# Corner params
# -----------------------------
PATTERN_SIZE = (7, 7)


@dataclass
class CornerParams:
    use_sb: bool = True
    classic_flags: int = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    expand_cells: float = 1.00
    extra_pad_px: int = 2
    max_expand_px: int = 300


def detect_corners_7x7(gray: np.ndarray, p: CornerParams) -> Optional[np.ndarray]:
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
    C = corners49.reshape(7, 7, 2)
    xs = C[:, :, 0]
    ys = C[:, :, 1]
    dx = np.diff(xs, axis=1).reshape(-1)
    dy = np.diff(ys, axis=0).reshape(-1)
    return float(np.median(np.abs(dx))), float(np.median(np.abs(dy)))


def compute_outer_crop_box(corners49: np.ndarray, img_w: int, img_h: int, p: CornerParams) -> Tuple[int, int, int, int]:
    cell_w, cell_h = estimate_cell_size_from_corners(corners49)

    min_x = float(np.min(corners49[:, 0]))
    max_x = float(np.max(corners49[:, 0]))
    min_y = float(np.min(corners49[:, 1]))
    max_y = float(np.max(corners49[:, 1]))

    ex = min(cell_w * p.expand_cells, float(p.max_expand_px))
    ey = min(cell_h * p.expand_cells, float(p.max_expand_px))

    l = int(np.floor(min_x - ex - p.extra_pad_px))
    r = int(np.ceil (max_x + ex + p.extra_pad_px))
    t = int(np.floor(min_y - ey - p.extra_pad_px))
    b = int(np.ceil (max_y + ey + p.extra_pad_px))

    l = max(0, l)
    t = max(0, t)
    r = min(img_w - 1, r)
    b = min(img_h - 1, b)

    if r <= l or b <= t:
        return (0, 0, img_w - 1, img_h - 1)

    return (l, t, r, b)


def crop_by_corners_or_none(roi_bgr: np.ndarray, p: CornerParams) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    corners = detect_corners_7x7(gray, p)
    if corners is None:
        return None
    H, W = gray.shape[:2]
    l, t, r, b = compute_outer_crop_box(corners, W, H, p)

    # ESKİ: return roi_bgr[t:b, l:r].copy()
    # YENİ (r,b inclusive clamp ediyorsun, slice exclusive olduğu için +1):
    return roi_bgr[t:b + 1, l:r + 1].copy()


# -----------------------------
# ROI candidate params
# -----------------------------
@dataclass
class RoiParams:
    min_component_area_ratio: float = 0.004
    max_component_area_ratio: float = 0.70
    min_wh: int = 110
    aspect_min: float = 0.50
    aspect_max: float = 2.20
    max_candidates_per_page: int = 120


def preprocess_binary_page(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,
                                31, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)


def find_candidate_rois(img_bgr: np.ndarray, p: RoiParams) -> List[Tuple[int, int, int, int]]:
    thr = preprocess_binary_page(img_bgr)
    H, W = thr.shape[:2]
    area_img = H * W
    num, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)

    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if w < p.min_wh or h < p.min_wh:
            continue
        if a < area_img * p.min_component_area_ratio:
            continue
        if a > area_img * p.max_component_area_ratio:
            continue
        aspect = w / float(h) if h else 0.0
        if not (p.aspect_min <= aspect <= p.aspect_max):
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:p.max_candidates_per_page]


# -----------------------------
# Hough params (dokunmuyoruz)
# -----------------------------
@dataclass
class HoughParams:
    out_size: int = 640
    canny1: int = 50
    canny2: int = 150
    hough_threshold: int = 60
    hough_min_len_ratio: float = 0.45
    hough_max_gap: int = 14
    line_angle_tol_deg: float = 12.0
    border_margin_ratio: float = 0.12
    border_pad_ratio: float = 0.006
    approx_eps_ratio: float = 0.02


def _angle_deg(x1, y1, x2, y2) -> float:
    ang = float(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
    return (ang + 180.0) % 180.0


def _line_len(x1, y1, x2, y2) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def _cluster_1d(values: List[int], weights: List[float], tol: int) -> List[Tuple[int, float]]:
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
            clusters.append((int(np.median(cur_vals)), cur_w))
            cur_vals = [int(v[i])]
            cur_w = float(w[i])

    clusters.append((int(np.median(cur_vals)), cur_w))
    return clusters


def _intersect_hv(hline: Tuple[int, int, int, int], vline: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, hline)
    x3, y3, x4, y4 = map(float, vline)

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return (np.nan, np.nan)

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
    return (px, py)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_by_grid_lines(roi_bgr: np.ndarray, p: HoughParams) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, p.canny1, p.canny2)

    h, w = edges.shape[:2]
    min_len = int(0.35 * min(h, w))

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(40, p.hough_threshold - 20),
        minLineLength=min_len,
        maxLineGap=max(10, p.hough_max_gap)
    )
    if lines is None:
        return None

    tol = p.line_angle_tol_deg

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

        if dy < dx and (ang <= tol or ang >= 180 - tol):
            y = int((y1 + y2) / 2)
            hy_vals.append(y); hy_w.append(L); hx_lines.append((x1, y1, x2, y2))

        if dx < dy and (abs(ang - 90) <= tol):
            x = int((x1 + x2) / 2)
            vx_vals.append(x); vx_w.append(L); vy_lines.append((x1, y1, x2, y2))

    if len(hy_vals) < 6 or len(vx_vals) < 6:
        return None

    tol_px = max(6, int(0.012 * min(h, w)))
    hy_clusters = _cluster_1d(hy_vals, hy_w, tol=tol_px)
    vx_clusters = _cluster_1d(vx_vals, vx_w, tol=tol_px)

    hy_clusters.sort(key=lambda t: t[1], reverse=True)
    vx_clusters.sort(key=lambda t: t[1], reverse=True)

    hy_sel = sorted([c[0] for c in hy_clusters[:9]])
    vx_sel = sorted([c[0] for c in vx_clusters[:9]])
    if len(hy_sel) < 6 or len(vx_sel) < 6:
        return None

    top_y = int(min(hy_sel))
    bot_y = int(max(hy_sel))
    left_x = int(min(vx_sel))
    right_x = int(max(vx_sel))

    def pick_nearest_h(y_target: int):
        best, bestd = None, 1e18
        for (x1, y1, x2, y2) in hx_lines:
            y = int((y1 + y2) / 2)
            d = abs(y - y_target)
            if d < bestd:
                bestd = d; best = (x1, y1, x2, y2)
        return best if bestd <= tol_px * 3 else None

    def pick_nearest_v(x_target: int):
        best, bestd = None, 1e18
        for (x1, y1, x2, y2) in vy_lines:
            x = int((x1 + x2) / 2)
            d = abs(x - x_target)
            if d < bestd:
                bestd = d; best = (x1, y1, x2, y2)
        return best if bestd <= tol_px * 3 else None

    h_top = pick_nearest_h(top_y)
    h_bot = pick_nearest_h(bot_y)
    v_left = pick_nearest_v(left_x)
    v_right = pick_nearest_v(right_x)
    if h_top is None or h_bot is None or v_left is None or v_right is None:
        return None

    tl = _intersect_hv(h_top, v_left)
    tr = _intersect_hv(h_top, v_right)
    br = _intersect_hv(h_bot, v_right)
    bl = _intersect_hv(h_bot, v_left)

    pts = np.array([tl, tr, br, bl], dtype=np.float32)
    if np.any(np.isnan(pts)):
        return None

    if not (0 <= pts[:, 0].min() < w and 0 <= pts[:, 0].max() < w and
            0 <= pts[:, 1].min() < h and 0 <= pts[:, 1].max() < h):
        return None

    pad = max(1, int(0.003 * min(w, h)))
    pts[0] += (pad, pad)
    pts[1] += (-pad, pad)
    pts[2] += (-pad, -pad)
    pts[3] += (pad, -pad)
    pts = order_points(pts)

    dst = np.array([[0, 0], [p.out_size - 1, 0], [p.out_size - 1, p.out_size - 1], [0, p.out_size - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(roi_bgr, M, (p.out_size, p.out_size))


def find_border_rect_in_roi(roi_bgr: np.ndarray, p: HoughParams) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, p.canny1, p.canny2)

    h, w = edges.shape[:2]
    min_len = int(p.hough_min_len_ratio * min(h, w))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=p.hough_threshold,
                            minLineLength=min_len,
                            maxLineGap=p.hough_max_gap)
    if lines is None:
        return None

    tol = p.line_angle_tol_deg
    margin_x = int(p.border_margin_ratio * w)
    margin_y = int(p.border_margin_ratio * h)

    horizontals: List[int] = []
    verticals: List[int] = []

    for x1, y1, x2, y2 in lines[:, 0]:
        ang = _angle_deg(x1, y1, x2, y2)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dy < dx and (ang <= tol or ang >= 180 - tol):
            y = int((y1 + y2) / 2)
            if y < margin_y or y > (h - margin_y):
                horizontals.append(y)

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

    pad = max(2, int(p.border_pad_ratio * min(w, h)))
    left = max(0, left + pad)
    right = min(w - 1, right - pad)
    top = max(0, top + pad)
    bottom = min(h - 1, bottom - pad)

    ww = right - left
    hh = bottom - top
    if ww <= 0 or hh <= 0:
        return None

    aspect = ww / float(hh)
    if not (0.80 <= aspect <= 1.28):
        return None

    return (left, top, right, bottom)


def fallback_warp_biggest_quad(roi_bgr: np.ndarray, p: HoughParams) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    # contours
    res = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, p.approx_eps_ratio * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(cnt)
        quad = cv2.boxPoints(rect).astype(np.float32)

    quad = order_points(quad)

    dst = np.array([[0, 0], [p.out_size - 1, 0], [p.out_size - 1, p.out_size - 1], [0, p.out_size - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(roi_bgr, M, (p.out_size, p.out_size))


# -----------------------------
# Classifier feature extraction (training ile birebir)
# -----------------------------
@dataclass
class ClfParams:
    img_size: Tuple[int, int] = (64, 64)
    proba_threshold: float = 0.50


def extract_features_for_clf(board_bgr: np.ndarray, p: ClfParams) -> np.ndarray:
    rgb = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize(p.img_size)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr.flatten().reshape(1, -1)


def clf_is_chessboard(clf, board_bgr: np.ndarray, p: ClfParams) -> Tuple[bool, float]:
    feat = extract_features_for_clf(board_bgr, p)

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(feat)[0]

        if hasattr(clf, "classes_"):
            classes = list(clf.classes_)
            score = float(probs[classes.index(1)])
        else:
            score = float(probs[1])

        return (score >= p.proba_threshold), score

    pred = int(clf.predict(feat)[0])
    return (pred == 1), float(pred)

def _box_area(l: int, t: int, r: int, b: int) -> int:
    # r,b inclusive
    return max(0, (r - l + 1)) * max(0, (b - t + 1))

def _is_near_fullpage_crop(
    l: int, t: int, r: int, b: int, W: int, H: int,
    min_cover_ratio: float = 0.55,
    max_margin_ratio: float = 0.08,
    aspect_min: float = 0.80,
    aspect_max: float = 1.25,
) -> bool:
    """
    Bu crop, sayfanın büyük kısmını kaplıyor mu ve yaklaşık kare mi?
    (Sayfanın zaten tek bir paddingsiz tahta olduğu senaryoyu yakalamak için.)
    """
    crop_area = _box_area(l, t, r, b)
    img_area = W * H
    if img_area <= 0:
        return False
    if crop_area / float(img_area) < min_cover_ratio:
        return False

    # margin'ler küçük mü?
    mx_l = l / float(W)
    mx_r = (W - 1 - r) / float(W)
    my_t = t / float(H)
    my_b = (H - 1 - b) / float(H)
    if max(mx_l, mx_r, my_t, my_b) > max_margin_ratio:
        return False

    ww = (r - l + 1)
    hh = (b - t + 1)
    if hh <= 0:
        return False
    asp = ww / float(hh)
    return (aspect_min <= asp <= aspect_max)


# -----------------------------
# Public API: page -> final boards
# -----------------------------
def extract_final_boards_from_page(
    page_bgr: np.ndarray,
    clf,
    roi_p: RoiParams,
    corner_p: CornerParams,
    hough_p: HoughParams,
    clf_p: ClfParams,
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Yeni sıra:
      0) FULL PAGE corner/grid kontrol
      1) ROI -> corner
      2) ROI -> hough/border/fallback -> clf
    """
    finals: List[Tuple[np.ndarray, str, float]] = []

    # ---------------------------------------------------------
    # 0) FULL PAGE: sayfa zaten paddingsiz/düzgün tahta mı?
    # ---------------------------------------------------------
    full_gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    full_corners = detect_corners_7x7(full_gray, corner_p)

    if full_corners is not None:
        H, W = full_gray.shape[:2]
        l, t, r, b = compute_outer_crop_box(full_corners, W, H, corner_p)

        if _is_near_fullpage_crop(l, t, r, b, W, H):
            full_crop = page_bgr[t:b + 1, l:r + 1].copy()
            finals.append((full_crop, "corners_fullpage", -1.0))

            # Bu noktada ROI'nin yanlış yer seçmesi problemi zaten çözülüyor.
            # Eğer "sayfada başka tahta da olabilir, ROI yine çalışsın" dersen
            # burada return etme; aşağı devam et.
            return finals

    # ---------------------------------------------------------
    # 1) ROI adaylarını bul
    # ---------------------------------------------------------
    rois = find_candidate_rois(page_bgr, roi_p)

    for (x, y, w, h) in rois:
        roi = page_bgr[y:y + h, x:x + w].copy()

        # 1a) ROI üstünde corner ile direkt düzgün crop dene
        corner_crop = crop_by_corners_or_none(roi, corner_p)
        if corner_crop is not None:
            finals.append((corner_crop, "corners", -1.0))
            continue

        # 2) corner yoksa: hough/border/fallback
        boards: List[np.ndarray] = []

        warped = warp_by_grid_lines(roi, hough_p)
        if warped is not None:
            boards.append(warped)
        else:
            border = find_border_rect_in_roi(roi, hough_p)
            if border is not None:
                l2, t2, r2, b2 = border
                bgr = roi[t2:b2, l2:r2].copy()
                bgr = cv2.resize(bgr, (hough_p.out_size, hough_p.out_size), interpolation=cv2.INTER_AREA)
                boards.append(bgr)
            else:
                fb = fallback_warp_biggest_quad(roi, hough_p)
                if fb is not None:
                    fb = cv2.resize(fb, (hough_p.out_size, hough_p.out_size), interpolation=cv2.INTER_AREA)
                    boards.append(fb)

        # 3) classifier
        for bgr in boards:
            ok, score = clf_is_chessboard(clf, bgr, clf_p)
            if ok:
                finals.append((bgr, "hough_clf", score))

    return finals