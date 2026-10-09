"""
mask1 inverse -> mask2 ile AND
out = mask2 & (~mask1)

- Geometriye DOKUNMAZ
- Boyut uyuşmazsa hata verir
- Pixel tipi değiştirmez
"""

from pathlib import Path
import numpy as np
from PIL import Image
import sys


# =============================
# KLASÖR YOLLARI
# =============================
MASK1_DIR = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V4_1\halo_mask_png")
MASK2_DIR = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\yedek\SiyahVeKaymaTemizMask")
OUTPUT_DIR = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V4_1\Sadece Traşlama V4_1 R2 Brain Mask")
# =============================


def read_exact(path: Path):
    img = Image.open(path)
    return np.array(img)


def main():

    if not MASK1_DIR.is_dir():
        raise NotADirectoryError(f"MASK1 klasör değil: {MASK1_DIR}")

    if not MASK2_DIR.is_dir():
        raise NotADirectoryError(f"MASK2 klasör değil: {MASK2_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask1_files = {p.name: p for p in MASK1_DIR.glob("*.png")}
    mask2_files = {p.name: p for p in MASK2_DIR.glob("*.png")}

    common_files = set(mask1_files.keys()) & set(mask2_files.keys())

    if not common_files:
        raise RuntimeError("Eşleşen dosya bulunamadı.")

    print(f"Eşleşen dosya sayısı: {len(common_files)}")

    for name in sorted(common_files):

        p1 = mask1_files[name]
        p2 = mask2_files[name]

        try:
            m1 = read_exact(p1)
            m2 = read_exact(p2)

            # ---- GEOMETRİ KONTROLÜ ----
            if m1.shape != m2.shape:
                raise ValueError(
                    f"{name} boyut uyumsuz!\n"
                    f"mask1: {m1.shape}\n"
                    f"mask2: {m2.shape}"
                )

            if m1.dtype != m2.dtype:
                raise ValueError(
                    f"{name} dtype uyumsuz!\n"
                    f"mask1: {m1.dtype}\n"
                    f"mask2: {m2.dtype}"
                )

            inv_m1 = np.bitwise_not(m1)
            out = np.bitwise_and(m2, inv_m1)

            out_path = OUTPUT_DIR / name
            Image.fromarray(out).save(out_path)

        except Exception as e:
            print(f"HATA ({name}): {e}")

    print("İşlem tamamlandı.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("GENEL HATA:")
        print(e)
        sys.exit(1)