# ocr_engine.py
import pytesseract
from PIL import Image, ImageOps
import os
import re
import platform


# --- 1. TESSERACT BULUCU (OTOMATİK) ---
def configure_tesseract():
    system_name = platform.system()
    if system_name == 'Windows':
        # Olası yollar listesi
        paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            os.path.expanduser(r'~\AppData\Local\Tesseract-OCR\tesseract.exe')
        ]
        for p in paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                return


configure_tesseract()


# --- 2. YARDIMCI FONKSİYONLAR ---

def add_padding(img, border=25):
    """
    Görselin etrafına beyaz çerçeve ekler. 
    Kenara yapışık yazıların (Örn: 'Mixed') okunmasını sağlar.
    """
    return ImageOps.expand(img, border=border, fill='white')


def is_garbage(text):
    """
    Gürültü Filtresi: Satranç tahtası, karmaşık grafikler gibi
    anlamsız çıktıları eler.
    """
    if not text: return True

    # Metni temizle
    text = text.strip()
    if len(text) < 3: return True

    # İzin verilen karakterler dışındakileri say (Harf, Sayı ve Temel Noktalama)
    # Satranç tahtasından gelen _ | \ / ~ gibi karakterleri yakalar.
    garbage_chars = re.sub(r'[a-zA-Z0-9çğıöşüÇĞİÖŞÜ.,:;\-\(\)\s"\'&]', '', text)

    # Eğer metnin %40'ından fazlası garip sembolse çöptür.
    if len(text) > 0:
        ratio = len(garbage_chars) / len(text)
        return ratio > 0.40
    return True


# --- 3. ANA FONKSİYON ---

def run_ocr(img):
    """
    Görseli analiz eder, gerekli ayarları yapar ve okur.
    """
    try:
        # A. Boyut Kontrolü (Çok küçük ikonları atla)
        if img.width < 20 or img.height < 10:
            return None

        # B. Kenar Boşluğu Ekle (Nefes Payı)
        img = add_padding(img)

        # C. Şerit Algılama (Aspect Ratio)
        # Eğer görsel çok geniş ve inceyse (Örn: Başlık şeridi)
        width, height = img.size
        custom_config = r'--oem 3 --psm 3'  # Varsayılan (Tam Sayfa)

        # Genişlik yüksekliğin 4 katından fazlaysa -> Tek Satır Modu (PSM 7)
        if width > height * 4:
            custom_config = r'--oem 3 --psm 7'

        # D. Okuma İşlemi
        text = pytesseract.image_to_string(img, lang='tur+eng', config=custom_config)
        clean_text = text.strip()

        # E. Çöp Kontrolü
        if is_garbage(clean_text):
            return None  # Çöp ise kaydetme (NULL olsun)

        return clean_text

    except Exception as e:
        print(f"!!! OCR HATASI !!!: {e}")
        return None