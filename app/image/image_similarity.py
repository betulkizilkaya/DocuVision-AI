import sqlite3
import io
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Set

import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import imagehash

from app.core.paths import DB_PATH


# ---------------------------
# DB
# ---------------------------
def create_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(DB_PATH),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------------------------
# MODELS
# ---------------------------
@dataclass(frozen=True)
class PdfImageRow:
    id: int
    file_id: int
    page_no: int
    image_index: int
    sha256: str          # render hash
    blob: bytes


# ---------------------------
# FETCH
# ---------------------------
def fetch_images(conn: sqlite3.Connection, *, file_id: Optional[int] = None) -> List[PdfImageRow]:
    cur = conn.cursor()
    if file_id is None:
        cur.execute("""
            SELECT id, file_id, page_no, image_index, sha256, blob
            FROM pdf_images
            ORDER BY file_id, page_no, image_index
        """)
    else:
        cur.execute("""
            SELECT id, file_id, page_no, image_index, sha256, blob
            FROM pdf_images
            WHERE file_id=?
            ORDER BY page_no, image_index
        """, (file_id,))

    out: List[PdfImageRow] = []
    for r in cur.fetchall():
        out.append(PdfImageRow(
            id=int(r["id"]),
            file_id=int(r["file_id"]),
            page_no=int(r["page_no"]),
            image_index=int(r["image_index"]),
            sha256=str(r["sha256"]),
            blob=bytes(r["blob"]),
        ))
    return out


def fetch_existing_pairs(conn: sqlite3.Connection) -> Set[tuple[int, int]]:
    cur = conn.cursor()
    cur.execute("SELECT image_id_a, image_id_b FROM image_similarity")
    return {(int(r[0]), int(r[1])) for r in cur.fetchall()}


# ---------------------------
# IMAGE IO + PREPROCESS
# ---------------------------
def pil_from_blob(blob: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
        return img
    except Exception:
        return None


def preprocess_gray_np(img_pil: Image.Image, *, target_max_side: int = 512) -> np.ndarray:
    img_gray = img_pil.convert("L")

    w, h = img_gray.size
    m = max(w, h)
    if m > target_max_side:
        scale = target_max_side / float(m)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_gray = img_gray.resize((new_w, new_h), Image.Resampling.BILINEAR)

    return np.array(img_gray, dtype=np.uint8)


# ---------------------------
# METRICS
# ---------------------------
def calc_ssim(img1: Image.Image, img2: Image.Image) -> float:
    try:
        i1 = preprocess_gray_np(img1)
        i2 = preprocess_gray_np(img2)

        h = min(i1.shape[0], i2.shape[0])
        w = min(i1.shape[1], i2.shape[1])
        if h < 7 or w < 7:
            return 0.0

        i1r = cv2.resize(i1, (w, h), interpolation=cv2.INTER_AREA)
        i2r = cv2.resize(i2, (w, h), interpolation=cv2.INTER_AREA)

        win_size = min(7, h, w)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            return 0.0

        score_normal, _ = ssim(i1r, i2r, full=True, win_size=win_size)
        i2_inv = 255 - i2r
        score_inverted, _ = ssim(i1r, i2_inv, full=True, win_size=win_size)

        s1 = 0.0 if np.isnan(score_normal) else float(score_normal)
        s2 = 0.0 if np.isnan(score_inverted) else float(score_inverted)
        return float(max(0.0, min(1.0, max(s1, s2))))
    except Exception:
        return 0.0


def calc_phash(img1: Image.Image, img2: Image.Image) -> float:
    try:
        h1 = imagehash.phash(img1)
        h2 = imagehash.phash(img2)
        return float(max(0.0, min(1.0, 1.0 - (h1 - h2) / 64.0)))
    except Exception:
        return 0.0


def _dynamic_feature_thresholds(img_np: np.ndarray) -> tuple[int, int]:
    """
    Görsel boyutuna göre minimum keypoint ve match eşiklerini ayarla.
    Amaç: küçük görsellerde (örn 94x94) feature'ın tamamen susmasını azaltmak,
    büyük görsellerde ise false-positive'i kontrol etmek.

    Returns:
      (min_kp, min_matches)
    """
    h, w = img_np.shape[:2]
    min_side = min(h, w)

    # 94x94 gibi küçük thumb görseller: daha düşük eşikler
    if min_side < 120:
        return 8, 4

    # orta boy: orta eşikler
    if min_side < 180:
        return 12, 6

    # büyük görseller: sıkı eşikler
    return 20, 10


def calc_orb(img1: Image.Image, img2: Image.Image) -> float:
    """
    ORB + CLAHE + crossCheck (dinamik eşikli sürüm)
    """
    try:
        i1 = preprocess_gray_np(img1)
        i2 = preprocess_gray_np(img2)
        if min(i1.shape) < 10 or min(i2.shape) < 10:
            return 0.0

        # Dinamik eşikler (iki görüntüden daha zayıf olana göre)
        mk1, mm1 = _dynamic_feature_thresholds(i1)
        mk2, mm2 = _dynamic_feature_thresholds(i2)
        min_kp = min(mk1, mk2)
        min_matches = min(mm1, mm2)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        i1 = clahe.apply(i1)
        i2 = clahe.apply(i2)

        orb = cv2.ORB_create(nfeatures=700)
        kp1, des1 = orb.detectAndCompute(i1, None)
        kp2, des2 = orb.detectAndCompute(i2, None)
        if des1 is None or des2 is None:
            return 0.0

        # --- FREN 1: minimum keypoint ---
        if min(len(kp1), len(kp2)) < min_kp:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches:
            return 0.0

        # --- FREN 2: minimum match ---
        if len(matches) < min_matches:
            return 0.0

        matches = sorted(matches, key=lambda x: x.distance)

        # top oranı küçük veride çok az kalmasın diye alt sınır koyuyoruz
        top_n = max(min_matches, int(len(matches) * 0.15))
        top = matches[:top_n]

        distances = [m.distance for m in top]
        if not distances:
            return 0.0

        sim = 1.0 - (float(np.mean(distances)) / 100.0)
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0


def calc_akaze(img1: Image.Image, img2: Image.Image) -> float:
    """
    AKAZE + Lowe ratio test (dinamik eşikli sürüm)
    """
    try:
        i1 = preprocess_gray_np(img1)
        i2 = preprocess_gray_np(img2)
        if min(i1.shape) < 10 or min(i2.shape) < 10:
            return 0.0

        # Dinamik eşikler (iki görüntüden daha zayıf olana göre)
        mk1, mm1 = _dynamic_feature_thresholds(i1)
        mk2, mm2 = _dynamic_feature_thresholds(i2)
        min_kp = min(mk1, mk2)
        min_good = min(mm1, mm2)  # AKAZE'de good match eşiği

        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(i1, None)
        kp2, des2 = akaze.detectAndCompute(i2, None)

        if des1 is None or des2 is None:
            return 0.0

        # --- FREN 1: minimum keypoint ---
        if min(len(kp1), len(kp2)) < min_kp:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in pairs:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # --- FREN 2: minimum good match ---
        if len(good) < min_good:
            return 0.0

        score = len(good) / float(min(len(kp1), len(kp2)))
        return float(max(0.0, min(1.0, score)))

    except Exception:
        return 0.0


# ---------------------------
# PAIRS (windowed within same PDF)
# ---------------------------
def iter_pairs_windowed(images: List[PdfImageRow], *, window: int = 5) -> Iterable[Tuple[PdfImageRow, PdfImageRow]]:
    n = len(images)
    for i in range(n):
        a = images[i]
        for j in range(i + 1, min(i + 1 + window, n)):
            b = images[j]
            if a.file_id != b.file_id:
                break
            yield a, b


# ---------------------------
# PIPELINE DECISION
# ---------------------------
def decide_pair(
    a_row: PdfImageRow,
    b_row: PdfImageRow,
    img_a: Image.Image,
    img_b: Image.Image,
    *,
    phash_gate: float = 0.85,
    phash_very_high: float = 0.90,   # <<< 0.90
    ssim_near: float = 0.92,
    ssim_mid_low: float = 0.85,
    ssim_similar_cut: float = 0.90,
    feature_gate: float = 0.15,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str, int, str]:
    """
    Returns:
      (ssim_v, phash_v, orb_v, akaze_v, label, decision_phase, reason)

    decision_phase:
      1 = exact hash
      2 = phash
      3 = ssim
      4 = feature
    """

    # --------------------------------------------------
    # Phase 1 — Exact duplicate (render hash)
    # --------------------------------------------------
    if a_row.sha256 and b_row.sha256 and a_row.sha256 == b_row.sha256:
        return (None, None, None, None, "EXACT_DUPLICATE", 1, "sha256")

    # --------------------------------------------------
    # Phase 2 — pHash candidate generation
    # --------------------------------------------------
    p = calc_phash(img_a, img_b)
    if p < phash_gate:
        return (None, float(p), None, None, "NOT_SIMILAR", 2, "phash_low")

    # --------------------------------------------------
    # Phase 3 — SSIM verification
    # --------------------------------------------------
    s = calc_ssim(img_a, img_b)

    # 3A — Very high SSIM → near duplicate
    if s >= ssim_near:
        return (float(s), float(p), None, None,
                "NEAR_DUPLICATE", 3, "phash+ssim")

    # 3B — SSIM very low band
    if s < ssim_mid_low:

        # --- GÜVENLİK FRENİ ---
        # SSIM < 0.50 ise override YAPMA
        if s < 0.50:
            return (float(s), float(p), None, None,
                    "NOT_SIMILAR", 3, "ssim_too_low_for_override")

        # --- pHash override (0.50 ≤ SSIM < 0.85 AND pHash ≥ 0.90) ---
        if p >= phash_very_high:
            o = calc_orb(img_a, img_b)
            ak = calc_akaze(img_a, img_b)

            if (o >= feature_gate) or (ak >= feature_gate):
                if o >= feature_gate and ak >= feature_gate:
                    return (float(s), float(p), float(o), float(ak),
                            "SIMILAR_THEME", 4, "phash_override+orb+akaze")
                if o >= feature_gate:
                    return (float(s), float(p), float(o), float(ak),
                            "SIMILAR_THEME", 4, "phash_override+orb")
                return (float(s), float(p), float(o), float(ak),
                        "SIMILAR_THEME", 4, "phash_override+akaze")

            return (float(s), float(p), float(o), float(ak),
                    "SIMILAR_THEME", 4, "phash_override_low_ssim")

        # --- normal düşük SSIM yolu ---
        return (float(s), float(p), None, None,
                "NOT_SIMILAR", 3, "phash+ssim_low")

    # --------------------------------------------------
    # Phase 4 — Ambiguous SSIM band → feature verification
    # --------------------------------------------------
    o = calc_orb(img_a, img_b)
    ak = calc_akaze(img_a, img_b)

    if (o >= feature_gate) or (ak >= feature_gate):
        if o >= feature_gate and ak >= feature_gate:
            return (float(s), float(p), float(o), float(ak),
                    "NEAR_DUPLICATE", 4, "ssim+orb+akaze")
        if o >= feature_gate:
            return (float(s), float(p), float(o), float(ak),
                    "NEAR_DUPLICATE", 4, "ssim+orb")
        return (float(s), float(p), float(o), float(ak),
                "NEAR_DUPLICATE", 4, "ssim+akaze")

    # --------------------------------------------------
    # Feature doğrulamadı → SSIM’e göre karar
    # --------------------------------------------------
    if s >= ssim_similar_cut:
        return (float(s), float(p), float(o), float(ak),
                "SIMILAR", 4, "ssim_high_feature_low")

    return (float(s), float(p), float(o), float(ak),
            "NOT_SIMILAR", 4, "ssim_feature_low")


# ---------------------------
# INSERT
# ---------------------------
def insert_similarity_batch(
    conn: sqlite3.Connection,
    rows: List[Tuple[int, int, Optional[float], Optional[float], Optional[float], Optional[float], str, int, str]],
) -> None:
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR IGNORE INTO image_similarity
        (image_id_a, image_id_b, ssim, phash, orb, akaze, label, decision_phase, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


# ---------------------------
# MAIN
# ---------------------------
def process(
    *,
    window: int = 5,
    batch_size: int = 250,
    file_id: Optional[int] = None,
    phash_gate: float = 0.85,
    ssim_near: float = 0.92,
    ssim_mid_low: float = 0.85,
    ssim_similar_cut: float = 0.90,
    feature_gate: float = 0.15,
) -> None:
    conn = create_connection()
    try:
        images = fetch_images(conn, file_id=file_id)
        print(f"[INFO] {len(images)} image(s) loaded. window={window}")

        existing = fetch_existing_pairs(conn)
        pending: List[Tuple[int, int, Optional[float], Optional[float], Optional[float], Optional[float], str, int, str]] = []

        for a_row, b_row in iter_pairs_windowed(images, window=window):
            ida, idb = (a_row.id, b_row.id)
            key = (min(ida, idb), max(ida, idb))
            if key in existing:
                continue

            img_a = pil_from_blob(a_row.blob)
            img_b = pil_from_blob(b_row.blob)
            if img_a is None or img_b is None:
                pending.append((key[0], key[1], None, None, None, None, "LOW_QUALITY", 2, "decode_fail"))
                existing.add(key)
                continue

            ssim_v, phash_v, orb_v, akaze_v, label, phase, reason = decide_pair(
                a_row, b_row, img_a, img_b,
                phash_gate=phash_gate,
                ssim_near=ssim_near,
                ssim_mid_low=ssim_mid_low,
                ssim_similar_cut=ssim_similar_cut,
                feature_gate=feature_gate,
            )

            pending.append((key[0], key[1], ssim_v, phash_v, orb_v, akaze_v, label, phase, reason))
            existing.add(key)

            if len(pending) >= batch_size:
                insert_similarity_batch(conn, pending)
                pending.clear()

            print(
                f"→ {key[0]}-{key[1]} | label={label} phase={phase} reason={reason} | "
                f"pHash={phash_v if phash_v is not None else 'NA'} "
                f"SSIM={ssim_v if ssim_v is not None else 'NA'} "
                f"ORB={orb_v if orb_v is not None else 'NA'} "
                f"AKAZE={akaze_v if akaze_v is not None else 'NA'}"
            )

        if pending:
            insert_similarity_batch(conn, pending)

        print("[✓] Pipeline similarity done.")
    finally:
        conn.close()


if __name__ == "__main__":
    process(window=5, batch_size=250)
