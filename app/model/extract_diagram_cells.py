import os
import cv2

BOARD_DIR = "models/deneme"
OUT_DIR = "diagram_cells_raw"
os.makedirs(OUT_DIR, exist_ok=True)

VALID_EXT = (".png", ".jpg", ".jpeg")

for fname in os.listdir(BOARD_DIR):
    if not fname.lower().endswith(VALID_EXT):
        continue

    path = os.path.join(BOARD_DIR, fname)
    img = cv2.imread(path)

    # --- UPSCALE (ZORUNLU) ---
    UPSCALE = 2.5  # burada 2.5 kullan
    img = cv2.resize(
        img,
        None,
        fx=UPSCALE,
        fy=UPSCALE,
        interpolation=cv2.INTER_CUBIC
    )

    h, w, _ = img.shape

    ch = h // 8
    cw = w // 8

    base = os.path.splitext(fname)[0]
    out_sub = os.path.join(OUT_DIR, base)
    os.makedirs(out_sub, exist_ok=True)

    for r in range(8):
        for c in range(8):
            cell = img[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            out_name = f"{base}_r{r}_c{c}.png"
            cv2.imwrite(os.path.join(out_sub, out_name), cell)

print("[OK] Diagram kareleri çıkarıldı.")
