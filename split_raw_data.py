import cv2
import numpy as np
import time
from pathlib import Path

# ---------------- MAKALE PARAMETRELERİ ----------------
BINARY_THRES = 220
KERNEL_SIZE = 20
MIN_AREA = 500
BETA = 10
GAMMA = 240
# ------------------------------------------------------


def threscontct_tbi(img):
    if img is None:
        return None

    # 1. Thresholding (α = 220)
    _, skull_thresh = cv2.threshold(img, BINARY_THRES, 255, cv2.THRESH_BINARY)

    # 2. Morphological Closing (20x20, ellipse)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
    )
    skull_closed = cv2.morphologyEx(
        skull_thresh, cv2.MORPH_CLOSE, kernel
    )

    # 3. Contour Filtering (A > 500)
    contours, _ = cv2.findContours(
        skull_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    skull_mask = np.zeros_like(img, dtype=np.uint8)

    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA:
            cv2.drawContours(
                skull_mask, [cnt], -1, color=255, thickness=-1
            )

    # 4. Distance Transform (β = 10)
    binary_skull = (skull_mask > 0).astype(np.uint8)
    dist_transform = cv2.distanceTransform(
        binary_skull, cv2.DIST_L2, 5
    )

    _, dist_mask = cv2.threshold(
        dist_transform, BETA, 255, cv2.THRESH_BINARY
    )
    final_mask = dist_mask.astype(np.uint8)

    # 5. Final Masking
    skull_stripped = cv2.bitwise_and(
        img, img, mask=final_mask
    )

    # 6. Gamma Cleaning (γ = 240)
    skull_stripped[skull_stripped > GAMMA] = 0

    return skull_stripped


def batch_process():
    # === SENİN VERDİĞİN DOSYA YOLLARI ===
    input_dir = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\SiyahVeKaymaTemizBrain")
    output_dir = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\SiyahVeKaymaTemizBrain_Algoritma_Cikti")
    # ===================================

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob("*.png"))
    total = len(image_files)

    if total == 0:
        print("❌ Girdi klasöründe görüntü yok.")
        return

    print(f"▶ Başlıyor: {total} görüntü işlenecek")
    print(f"▶ Çıktı klasörü: {output_dir}")
    print("-" * 50)

    start_time = time.time()
    success = 0
    fail = 0

    for i, img_path in enumerate(image_files, start=1):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            result = threscontct_tbi(img)

            if result is not None:
                save_path = output_dir / img_path.name
                cv2.imwrite(str(save_path), result)
                success += 1
            else:
                fail += 1

        except Exception as e:
            print(f"Hata ({img_path.name}): {e}")
            fail += 1

        if i % 100 == 0 or i == total:
            print(f"İlerleme: {i}/{total}")

    end_time = time.time()
    duration = end_time - start_time

    print("-" * 50)
    print("✅ İŞLEM TAMAMLANDI")
    print(f"Başarılı : {success}")
    print(f"Hatalı   : {fail}")
    print(f"Toplam süre : {duration:.2f} sn")
    print(f"Ortalama / görüntü : {duration/total:.4f} sn")
    print("📌 Makalede bildirilen ortalama: ~0.62 sn / görüntü")


if __name__ == "__main__":
    batch_process()