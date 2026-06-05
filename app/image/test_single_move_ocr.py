import cv2
from app.image.chess_notation_ocr import preprocess_for_ocr, run_tesseract_ocr

img = cv2.imread(r"D:\GithubProject\ProjectNexus-Intelligent-PDF-Analysis\temp\test_move.png")

prep = preprocess_for_ocr(img)
text = run_tesseract_ocr(prep)

print("RAW OCR:", repr(text))

cv2.imwrite(
    r"D:\GithubProject\ProjectNexus-Intelligent-PDF-Analysis\temp\test_move_prep.png",
    prep
)