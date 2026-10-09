import os
import cv2
import numpy as np
from pathlib import Path

# ==============================
# BURAYA KLASÖR YOLUNU GİR
# ==============================
IMAGE_DIR = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Algoritma_Brain_Mask")
OUTPUT_TXT = "siyah_resimler.txt"


def imread_unicode(path):
    """
    Türkçe karakter içeren yollar için güvenli okuma
    """
    stream = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    return img


def main():
    if not IMAGE_DIR.exists():
        print("Klasör bulunamadı.")
        return

    black_images = []

    files = sorted(IMAGE_DIR.glob("*.png"))  # png değilse uzantıyı değiştir

    print(f"Toplam {len(files)} dosya bulundu.\n")

    for file in files:
        img = imread_unicode(file)

        if img is None:
            print(f"Okunamadı: {file.name}")
            continue

        # Eğer maksimum piksel değeri 0 ise tamamen siyahtır
        if np.max(img) == 0:
            black_images.append(file.name)

    # txt kaydet
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for name in black_images:
            f.write(name + "\n")

    print("="*40)
    print(f"Simsiyah bulunan dosya sayısı: {len(black_images)}")
    print(f"Sonuçlar '{OUTPUT_TXT}' dosyasına kaydedildi.")
    print("="*40)


if __name__ == "__main__":
    main()