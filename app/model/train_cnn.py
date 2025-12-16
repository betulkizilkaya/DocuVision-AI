import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import matplotlib.pyplot as plt

# --- AYARLAR (HIPERPARAMETRELER) ---
# Windows uyumlu hale getirdiğimiz klasörün yolu
DATASET_PATH = "dataset_windows/train"

# Modelin ve etiketlerin kaydedileceği yollar
MODEL_SAVE_PATH = "models/chess_model_v2.h5"
CLASS_INDICES_PATH = "models/class_indices.json"

# Neden 32x32? Taşların şeklini anlamak için yeterli ve hızlı.
IMG_SIZE = (32, 32)
# Neden 32? İdeal öğrenme hızı ve bellek kullanımı için standart.
BATCH_SIZE = 32
# Neden 15? Veri boyutuna göre modelin öğrenmesi için yeterli süre.
EPOCHS = 15

# Modeller klasörü yoksa oluştur
if not os.path.exists("models"):
    os.makedirs("models")

# --- 1. VERİ YÜKLEME VE İŞLEME ---
# Resim piksellerini 0-255'ten 0-1 arasına sıkıştırıyoruz (Normalizasyon)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Verinin %20'sini test (sınav) için ayırıyoruz
)

print(f"[INFO] '{DATASET_PATH}' klasöründen veriler okunuyor...")

try:
    # Eğitim Seti (Modelin ders çalışacağı kitap)
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        color_mode='grayscale', # Renk gürültüsünü atmak için gri yapıyoruz
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Doğrulama Seti (Modelin sınav olacağı kitap)
    validation_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
except Exception as e:
    print(f"HATA: Veri okunamadı. Klasör yolunu kontrol et: {DATASET_PATH}")
    exit()

# Sınıf İsimlerini Kaydet (Black_King -> 0, White_Pawn -> 1 gibi)
class_indices = train_generator.class_indices
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(class_indices, f)
print(f"[INFO] Tespit edilen sınıflar ({len(class_indices)} adet):")
print(list(class_indices.keys()))

# --- 2. CNN MODEL MİMARİSİ ---
model = Sequential([
    # Giriş Katmanı
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

    # 1. Özellik Çıkarma Bloğu (Kenarları bulur)
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # 2. Özellik Çıkarma Bloğu (Şekilleri bulur)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # 3. Özellik Çıkarma Bloğu (Taşları tanır)
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Karar Verme Bloğu (Sınıflandırma)
    Flatten(), # 2D resmi düz bir listeye çevirir
    Dense(128, activation='relu'), # Düşünen katman
    Dropout(0.5), # Ezberlemeyi önlemek için %50 unutma uygular
    Dense(len(class_indices), activation='softmax') # Sonuç katmanı (Olasılıkları verir)
])

# Modeli Derle
model.compile(optimizer='adam', # Akıllı öğrenme algoritması
              loss='categorical_crossentropy', # Çok sınıflı hata hesaplama yöntemi
              metrics=['accuracy'])

model.summary()

# --- 3. EĞİTİMİ BAŞLAT ---
print("\n[INFO] Eğitim başlıyor... (Epoch 1/15)")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- 4. KAYDETME VE RAPORLAMA ---
model.save(MODEL_SAVE_PATH)
print(f"\n[BAŞARILI] Model kaydedildi: {MODEL_SAVE_PATH}")

# Grafik Çiz (Raporun için gerekli)
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Eğitim Başarısı (Train)')
plt.plot(history.history['val_accuracy'], label='Test Başarısı (Validation)')
plt.title('Satranç Taşı Tanıma Başarısı')
plt.xlabel('Epoch (Tur)')
plt.ylabel('Doğruluk (Accuracy)')
plt.legend()
plt.grid(True)
plt.savefig("models/basari_grafigi.png")
print("[INFO] Grafik 'models/basari_grafigi.png' dosyasına kaydedildi.")