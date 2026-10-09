"""
Ana makalenin algoritmasına göre beyinler JHO veri setine göre çıkartıldı (skull_stripped_JHO) ve binary maskeleri elde edildi
"""

import os
from pathlib import Path
import cv2
import numpy as np

# =============================
# PATHS (senin yolların)
# =============================
INPUT_DIR   = Path(r"C:\Users\Yusuf Soylu\Desktop\JHO Beyaz alan silme deneme 1069\Temiz_Beyin_R2\cleaned_png")
OUT_ALL     = Path(r"C:\Users\Yusuf Soylu\Desktop\JHO Beyaz alan silme deneme 1069\Temiz_Beyin_R2\uzman_2_1069_Beyaz_alan temizlenmis_all")
OUT_LARGEST = Path(r"C:\Users\Yusuf Soylu\Desktop\JHO Beyaz alan silme deneme 1069\Temiz_Beyin_R2\uzman_2_1069_Beyaz_alan temizlenmis_largest")

OUT_ALL.mkdir(parents=True, exist_ok=True)
OUT_LARGEST.mkdir(parents=True, exist_ok=True)

PIXEL_THR = 3
EMPTY_FRAC_THR = 0.0005
DO_CLOSING = True
CLOSING_K = 5

def safe_imwrite(path: Path, img: np.ndarray):
    ok = cv2.imwrite(str(path), img)
    if not ok:
        print(f"[HATA] Yazılamadı: {path}")
    return ok

def fill_holes(mask255: np.ndarray) -> np.ndarray:
    h, w = mask255.shape
    flood = mask255.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask255, flood_inv)

def largest_component(mask01: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    if num <= 1:
        return (mask01 * 255).astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]  # background hariç
    best = 1 + np.argmax(areas)
    return (labels == best).astype(np.uint8) * 255

def process_one(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("[UYARI] Okunamadı:", img_path)
        return

    fg = (img > PIXEL_THR).astype(np.uint8)

    if fg.mean() < EMPTY_FRAC_THR:
        empty = np.zeros_like(img, dtype=np.uint8)
        safe_imwrite(OUT_ALL / img_path.name, empty)
        safe_imwrite(OUT_LARGEST / img_path.name, empty)
        return

    mask_all = (fg * 255).astype(np.uint8)
    mask_big = largest_component(fg)

    if DO_CLOSING:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_K, CLOSING_K))
        mask_big = cv2.morphologyEx(mask_big, cv2.MORPH_CLOSE, k)

    mask_big = fill_holes(mask_big)

    safe_imwrite(OUT_ALL / img_path.name, mask_all)
    safe_imwrite(OUT_LARGEST / img_path.name, mask_big)

def main():
    if not INPUT_DIR.exists():
        print("[HATA] INPUT_DIR yok:", INPUT_DIR)
        return

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # >>> ALT KLASÖRLERİ DE GEZ <<< (en kritik fark bu)
    files = sorted([p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    print("INPUT_DIR:", INPUT_DIR)
    print("Bulunan görüntü sayısı:", len(files))
    if len(files) == 0:
        print("Hiç dosya bulamadım. Klasörün içinde PNG'ler alt klasörde mi? veya uzantı farklı mı?")
        return

    for i, p in enumerate(files, 1):
        process_one(p)
        if i % 100 == 0:
            print(f"{i}/{len(files)} işlendi")

    print("Bitti.")
    print("OUT_ALL     :", OUT_ALL, "->", len(list(OUT_ALL.glob('*.png'))), "png")
    print("OUT_LARGEST :", OUT_LARGEST, "->", len(list(OUT_LARGEST.glob('*.png'))), "png")

if __name__ == "__main__":
    main()
