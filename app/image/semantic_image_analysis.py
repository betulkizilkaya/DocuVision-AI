import sqlite3
import io
import re
import csv
import cv2
import numpy as np
import chess

from PIL import Image
from pathlib import Path
from ultralytics import YOLO

from app.core.paths import DB_PATH

print("SEMANTIC IMAGE ANALYSIS FILE LOADED")


# ---------------------------
# CONFIG
# ---------------------------
PERSON_MODEL_PATH = "yolov8s.pt"
PERSON_CONF_THRESHOLD = 0.45
PERSON_MIN_BOX_AREA_RATIO = 0.01
SAVE_PERSON_DEBUG_BOXES = True

LOGO_SCORE_THRESHOLD = 0.62
LOGO_TEXT_PENALTY_LONG = 0.45
LOGO_TEXT_PENALTY_MEDIUM = 0.70
LOGO_PERSON_PENALTY = 0.55
LOGO_PORTRAIT_LAYOUT_PENALTY = 0.65
LOGO_CENTER_FRAME_PENALTY = 0.65

# Dominance thresholds
PERSON_DOMINANCE_MIN_AREA = 0.18
PERSON_DOMINANCE_MIN_SCORE = 0.55

LOGO_DOMINANCE_MIN_SCORE = 0.68
LOGO_DOMINANCE_MIN_AREA = 0.015
LOGO_DOMINANCE_MAX_AREA = 0.22

TEXT_HEAVY_THRESHOLD = 80


# ---------------------------
# DB
# ---------------------------
def create_connection():
    return sqlite3.connect(str(DB_PATH))


def fetch_images(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, p.blob
        FROM pdf_images p
        INNER JOIN image_features f ON p.id = f.image_id
        WHERE f.is_chessboard = 0
        ORDER BY p.file_id, p.page_no, p.image_index
    """)
    return cur.fetchall()


# ---------------------------
# PERSON DETECTION (YOLOv8)
# ---------------------------
print(f"[INFO] Loading YOLO person model: {PERSON_MODEL_PATH}")
person_model = YOLO(PERSON_MODEL_PATH)


def detect_person(img):
    """
    Returns:
        has_person (0/1)
        person_score (0.0-1.0)
        person_boxes [(x1,y1,x2,y2,conf), ...]
        max_person_area_ratio
    """
    try:
        img_cv = np.array(img)

        if img_cv is None or img_cv.size == 0:
            return 0, 0.0, [], 0.0

        if len(img_cv.shape) == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

        if img_cv.dtype != np.uint8:
            img_cv = img_cv.astype(np.uint8)

        img_cv = np.ascontiguousarray(img_cv)

        h, w = img_cv.shape[:2]
        if h <= 0 or w <= 0:
            return 0, 0.0, [], 0.0

        image_area = float(h * w)

        results = person_model.predict(
            source=img_cv,
            conf=PERSON_CONF_THRESHOLD,
            verbose=False
        )

        person_boxes = []
        best_score = 0.0
        max_area_ratio = 0.0

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                box_area = float(bw * bh)

                if image_area <= 0:
                    continue

                area_ratio = box_area / image_area

                if area_ratio < PERSON_MIN_BOX_AREA_RATIO:
                    continue

                person_boxes.append((x1, y1, x2, y2, round(conf, 4)))
                best_score = max(best_score, conf)
                max_area_ratio = max(max_area_ratio, area_ratio)

        if person_boxes:
            return 1, round(min(best_score, 1.0), 4), person_boxes, round(max_area_ratio, 4)

        return 0, 0.0, [], 0.0

    except Exception as e:
        print(f"[PERSON ERROR] {e}")
        return 0, 0.0, [], 0.0


def draw_person_boxes(img, person_boxes):
    img_cv = np.array(img).copy()

    if len(img_cv.shape) == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

    for (x1, y1, x2, y2, conf) in person_boxes:
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_cv,
            f"person {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return Image.fromarray(img_cv)


# ---------------------------
# LOGO DETECTION
# ---------------------------
def safe_entropy(gray_img):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.astype(np.float32)
    total = hist.sum()

    if total <= 0:
        return 0.0

    hist /= total
    return float(-np.sum(hist * np.log2(hist + 1e-7)))


def get_ocr_text(conn, image_id):
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT text_raw FROM ocr_extracts WHERE image_id=?",
            (image_id,)
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return ""
        return row[0].strip()
    except Exception as e:
        print(f"[OCR TEXT ERROR] image_id={image_id}, error={e}")
        return ""


def estimate_text_density(text):
    if not text:
        return 0.0

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return 0.0

    alnum_count = sum(ch.isalnum() for ch in cleaned)
    return alnum_count / max(len(cleaned), 1)


def detect_logo_heuristic(img, ocr_text="", person=0, p_score=0.0):
    """
    Returns:
        has_logo
        logo_score
        logo_dominance_area_ratio
    """
    try:
        img_cv = np.array(img)

        if img_cv is None or img_cv.size == 0:
            return 0, 0.0, 0.0

        if len(img_cv.shape) == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

        h, w = img_cv.shape[:2]
        area = h * w

        if area <= 0:
            return 0, 0.0, 0.0

        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = float(np.sum(edges > 0)) / float(area)

        small_for_color = img_cv
        if max(h, w) > 256:
            scale = 256.0 / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            small_for_color = cv2.resize(
                img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

        pixels = small_for_color.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))
        color_ratio = unique_colors / max(len(pixels), 1)

        entropy = safe_entropy(gray)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        large_contours = [c for c in contours if cv2.contourArea(c) > area * 0.01]
        medium_contours = [c for c in contours if cv2.contourArea(c) > area * 0.002]

        contour_count = len(large_contours)
        medium_contour_count = len(medium_contours)

        compact_score = 0.0
        largest_rect = None
        largest_rect_area = 0.0

        if large_contours:
            areas = [cv2.contourArea(c) for c in large_contours]
            max_area = max(areas)
            compact_score = float(max_area) / float(area)

            for c in large_contours:
                x, y, cw, ch = cv2.boundingRect(c)
                rect_area = cw * ch
                if rect_area > largest_rect_area:
                    largest_rect_area = rect_area
                    largest_rect = (x, y, cw, ch)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )

        component_areas = []
        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            if area * 0.001 < comp_area < area * 0.2:
                component_areas.append(comp_area)

        component_count = len(component_areas)
        aspect = w / float(h) if h > 0 else 1.0

        # Base score
        score = 0.0

        if 0.01 < edge_ratio < 0.12:
            score += 0.22
        elif 0.12 <= edge_ratio < 0.18:
            score += 0.10

        if color_ratio < 0.12:
            score += 0.18
        elif color_ratio < 0.20:
            score += 0.08

        if 2.3 < entropy < 5.2:
            score += 0.18
        elif 2.0 < entropy < 5.8:
            score += 0.08

        if 1 <= contour_count <= 8:
            score += 0.14
        elif 1 <= medium_contour_count <= 15:
            score += 0.06

        if 0.015 < compact_score < 0.35:
            score += 0.12
        elif 0.01 < compact_score < 0.45:
            score += 0.05

        if 2 <= component_count <= 18:
            score += 0.11
        elif 1 <= component_count <= 28:
            score += 0.05

        if 0.6 <= aspect <= 3.2:
            score += 0.05

        # Penalties
        cleaned_text = re.sub(r"\s+", " ", ocr_text).strip()
        text_len = len(cleaned_text)
        text_density = estimate_text_density(cleaned_text)

        if text_len >= 80 and text_density > 0.55:
            score *= LOGO_TEXT_PENALTY_LONG
        elif text_len >= 35 and text_density > 0.45:
            score *= LOGO_TEXT_PENALTY_MEDIUM

        if person == 1 and p_score >= 0.40:
            score *= LOGO_PERSON_PENALTY

        if h > w * 1.35:
            score *= LOGO_PORTRAIT_LAYOUT_PENALTY

        logo_area_ratio = 0.0

        if largest_rect is not None:
            x, y, rw, rh = largest_rect
            rect_area_ratio = (rw * rh) / float(area)
            logo_area_ratio = rect_area_ratio

            center_x = x + rw / 2.0
            center_y = y + rh / 2.0

            near_center = (
                abs(center_x - (w / 2.0)) < w * 0.18 and
                abs(center_y - (h / 2.0)) < h * 0.18
            )

            tall_rect = rh > rw * 1.10

            if near_center and tall_rect and 0.18 < rect_area_ratio < 0.70:
                score *= LOGO_CENTER_FRAME_PENALTY

        final_score = round(min(max(score, 0.0), 1.0), 4)

        if final_score >= LOGO_SCORE_THRESHOLD:
            return 1, final_score, round(logo_area_ratio, 4)

        return 0, final_score, round(logo_area_ratio, 4)

    except Exception as e:
        print(f"[LOGO HEURISTIC ERROR] {e}")
        return 0, 0.0, 0.0


# ---------------------------
# GAME NOTATION
# ---------------------------
def detect_game_notation(conn, image_id):
    try:
        text = get_ocr_text(conn, image_id)

        if not text:
            return 0, 0.0

        text = normalize_chess_ocr(text)

        tokens = chess_pattern.findall(text)
        if not tokens:
            return 0, 0.0

        move_numbers = 0
        san_like_moves = 0
        valid_san_parses = 0

        # 1) Güçlü yüzey sinyalleri
        for token in tokens:
            token = token.strip()

            if re.fullmatch(r"\d+\.(?:\.\.)?", token):
                move_numbers += 1
            else:
                san_like_moves += 1

        # 2) Tek lineer oyun gibi değil, bağımsız aday hamle olarak doğrula
        # Her token için başlangıç tahtasında parse denemesi:
        # Amaç tam oyunu kurmak değil, token'ların gerçekten SAN benzeri olup olmadığını görmek
        for token in tokens:
            token = token.strip()

            if re.fullmatch(r"\d+\.(?:\.\.)?", token):
                continue

            try:
                board = chess.Board()
                board.parse_san(token)
                valid_san_parses += 1
            except Exception:
                pass

        total_signal = (
            move_numbers * 1.0 +
            san_like_moves * 1.2 +
            valid_san_parses * 1.5
        )

        # Satranç dergisi / analiz sayfaları için daha uygun eşik
        if (move_numbers >= 3 and san_like_moves >= 8) or total_signal >= 18:
            score = min(1.0, total_signal / 30.0)
            return 1, round(score, 4)

        return 0, 0.0

    except Exception as e:
        print(f"[GAME ERROR] image_id={image_id}, error={e}")
        return 0, 0.0

def normalize_chess_ocr(text):
    replacements = {
        "0-0-0": "O-O-O",
        "0-0": "O-O",
        "o-o-o": "O-O-O",
        "o-o": "O-O",
        "×": "x",
        "–": "-",
        "—": "-",
        "♔": "K", "♕": "Q", "♖": "R", "♗": "B", "♘": "N",
        "♚": "K", "♛": "Q", "♜": "R", "♝": "B", "♞": "N",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


chess_pattern = re.compile(
    r"""
    (O-O-O|O-O|
    [KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|
    [a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|
    [a-h][1-8](?:=[QRBN])?[+#]?|
    \d+\.(?:\.\.)?)
    """,
    re.VERBOSE
)


# ---------------------------
# DOMINANCE HELPERS
# ---------------------------
def is_person_dominant(person, p_score, person_area_ratio, ocr_text):
    text_len = len(re.sub(r"\s+", " ", ocr_text).strip())

    if not person:
        return False

    if p_score < PERSON_DOMINANCE_MIN_SCORE:
        return False

    if person_area_ratio < PERSON_DOMINANCE_MIN_AREA:
        return False

    # Çok yoğun metin varsa bu genelde afiş/kapak/reklamdır
    if text_len >= TEXT_HEAVY_THRESHOLD:
        return False

    return True


def is_logo_dominant(logo, l_score, logo_area_ratio, ocr_text, person, person_area_ratio):
    text_len = len(re.sub(r"\s+", " ", ocr_text).strip())

    if not logo:
        return False

    # 🔥 FULL LOGO OVERRIDE (EN KRİTİK EKLEME)
    if l_score >= 0.75 and logo_area_ratio >= 0.25:
        return True

    if l_score < LOGO_DOMINANCE_MIN_SCORE:
        return False

    # Çok küçük → köşe logosu
    if logo_area_ratio < LOGO_DOMINANCE_MIN_AREA:
        return False

    # 🔥 BU SATIRI YUMUŞATIYORUZ
    if logo_area_ratio > 0.5:
        return True  # büyükse direkt logo

    # Çok yazı → sadece küçük logolarda ceza ver
    if text_len >= TEXT_HEAVY_THRESHOLD and logo_area_ratio < 0.15:
        return False

    if person == 1 and person_area_ratio >= 0.15:
        return False

    return True


# ---------------------------
# LABEL SELECTION
# ---------------------------
def choose_label(
    person, p_score,
    logo, l_score,
    game, g_score,
    person_area_ratio=0.0,
    logo_area_ratio=0.0,
    ocr_text=""
):
    person_dom = is_person_dominant(person, p_score, person_area_ratio, ocr_text)
    logo_dom = is_logo_dominant(logo, l_score, logo_area_ratio, ocr_text, person, person_area_ratio)

    if game == 1 and g_score >= 0.45:
        return "game_notation", round(g_score, 4)

    if person_dom and (p_score >= l_score):
        return "person", round(p_score, 4)

    if logo_dom and (l_score > p_score):
        return "logo", round(l_score, 4)

    # Dominant değilse unknown
    return "unknown", 0.0


# ---------------------------
# UPDATE / INSERT DB
# ---------------------------
def update_features(conn, image_id, data):
    cur = conn.cursor()

    exists = cur.execute(
        "SELECT 1 FROM image_features WHERE image_id=? LIMIT 1",
        (image_id,)
    ).fetchone()

    if image_id <= 3:
        print(f"update_features called for image_id={image_id}, exists={bool(exists)}")

    if exists:
        cur.execute("""
            UPDATE image_features SET
            has_person=?,
            person_score=?,
            has_logo=?,
            logo_score=?,
            has_game_notation=?,
            game_notation_score=?,
            predicted_label=?,
            predicted_confidence=?
            WHERE image_id=?
        """, (*data, image_id))
    else:
        cur.execute("""
            INSERT INTO image_features (
                image_id,
                has_person,
                person_score,
                has_logo,
                logo_score,
                has_game_notation,
                game_notation_score,
                predicted_label,
                predicted_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (image_id, *data))

    if image_id <= 3:
        print(f"rowcount={cur.rowcount}, data={data}")


# ---------------------------
# MAIN
# ---------------------------
def process_all():
    print("PROCESS_ALL STARTED")

    conn = create_connection()
    images = fetch_images(conn)
    print(f"[CHECK] Gelen görsel sayısı: {len(images)}")
    print(f"{len(images)} görsel analiz ediliyor...")

    out_dir = Path("semantic_labeled_images")
    out_dir.mkdir(exist_ok=True)

    debug_person_dir = out_dir / "person_debug"
    if SAVE_PERSON_DEBUG_BOXES:
        debug_person_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "semantic_results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "image_id",
        "person", "person_score",
        "logo", "logo_score",
        "game", "game_score",
        "label", "confidence",
        "saved_image",
        "person_box_count",
        "person_area_ratio",
        "logo_area_ratio",
        "ocr_text_len"
    ])

    try:
        for i, (img_id, blob) in enumerate(images, 1):
            if i <= 5:
                print(f"[DEBUG] image_id={img_id}")

            try:
                img = Image.open(io.BytesIO(blob)).convert("RGB")

                person, p_score, person_boxes, person_area_ratio = detect_person(img)
                ocr_text = get_ocr_text(conn, img_id)
                logo, l_score, logo_area_ratio = detect_logo_heuristic(
                    img,
                    ocr_text=ocr_text,
                    person=person,
                    p_score=p_score
                )
                game, g_score = detect_game_notation(conn, img_id)

                label, conf = choose_label(
                    person, p_score,
                    logo, l_score,
                    game, g_score,
                    person_area_ratio=person_area_ratio,
                    logo_area_ratio=logo_area_ratio,
                    ocr_text=ocr_text
                )

                if i <= 10:
                    print(
                        f"image_id={img_id}, "
                        f"person={person}, p_score={p_score}, boxes={len(person_boxes)}, "
                        f"person_area_ratio={person_area_ratio}, "
                        f"logo={logo}, l_score={l_score}, logo_area_ratio={logo_area_ratio}, "
                        f"game={game}, g_score={g_score}, "
                        f"text_len={len(ocr_text)}, "
                        f"label={label}, conf={conf}"
                    )

                update_features(
                    conn,
                    img_id,
                    (person, p_score, logo, l_score, game, g_score, label, conf)
                )

                safe_label = str(label).replace(" ", "_")
                safe_conf = f"{conf:.4f}"
                image_filename = f"{img_id}_{safe_label}_{safe_conf}.png"
                image_path = out_dir / image_filename

                img.save(image_path)

                if SAVE_PERSON_DEBUG_BOXES and person_boxes:
                    debug_img = draw_person_boxes(img, person_boxes)
                    debug_img.save(debug_person_dir / f"{img_id}_person_boxes.png")

                csv_writer.writerow([
                    img_id,
                    person, p_score,
                    logo, l_score,
                    game, g_score,
                    label, conf,
                    image_filename,
                    len(person_boxes),
                    person_area_ratio,
                    logo_area_ratio,
                    len(ocr_text)
                ])

                if i % 50 == 0:
                    conn.commit()
                    print(f"{i} işlendi ve commit edildi")

            except Exception as e:
                print(f"[ERROR] image_id={img_id} hata={e}")

        print("LOOP FINISHED, COMMITTING...")
        conn.commit()
        print("COMMIT DONE")

    finally:
        csv_file.close()
        conn.close()


if __name__ == "__main__":
    process_all()