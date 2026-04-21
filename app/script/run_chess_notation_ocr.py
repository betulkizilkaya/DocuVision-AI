from app.image.chess_notation_ocr import process_single_image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def main():
    # test için elle ver
    image_id = 1
    image_path = r"C:\Users\betul\OneDrive\Belgeler\GitHub\DocuVision-AI\temp\test_notation.png"

    process_single_image(image_id, image_path)
    print("[OK] OCR + SAN doğrulama tamamlandı.")

if __name__ == "__main__":
    main()