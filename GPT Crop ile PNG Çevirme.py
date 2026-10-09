from pathlib import Path
import re

from PIL import Image
import nibabel as nib
import numpy as np
import cv2


# ==========================
# PATHS (BURAYI DÜZENLE)
# ==========================
CT_DIR = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\orijinal_CT")
R2_DIR = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Manual_Mask_Reader_2")

OUT_BASE = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\PNG data")
OUT_IMG  = OUT_BASE / "image"
OUT_LBL  = OUT_BASE / "label_r2"

# Window (Makale ile aynı başlangıç)
WL = 40.0
WW = 120.0

# Slice isimleri 0001'den mi başlasın?
ONE_BASED_SLICE = True

# Mask boş sayılmasın diye minimum piksel
MIN_MASK_PIXELS = 50


# ==========================
# 1) NIfTI OKUMA
# ==========================
def load_nifti(path: Path):
    img = nib.load(str(path))
    vol = img.get_fdata()
    affine = img.affine
    spacing = img.header.get_zooms()[:3]
    return vol, affine, spacing


# ==========================
# 2) WINDOW -> UINT8
# ==========================
def window_to_uint8(vol_hu: np.ndarray, level: float = 40.0, width: float = 120.0) -> np.ndarray:
    low = level - (width / 2.0)
    high = level + (width / 2.0)

    vol_clipped = np.clip(vol_hu, low, high)
    vol_norm = (vol_clipped - low) / (high - low)
    vol_255 = vol_norm * 255.0
    return vol_255.astype(np.uint8)


# ==========================
# 3) MASK'TEN MEDIAN MERKEZ
# ==========================
def slice_bbox_center_y(mask2d: np.ndarray, min_pixels: int = 50):
    m = (mask2d > 0)
    if m.sum() < min_pixels:
        return None
    ys, xs = np.where(m)
    x_min = int(xs.min())
    x_max = int(xs.max())
    return (x_min + x_max) // 2


def volume_median_center_y(mask3d: np.ndarray, min_pixels: int = 50):
    centers = []
    for s in range(mask3d.shape[2]):
        c = slice_bbox_center_y(mask3d[:, :, s], min_pixels=min_pixels)
        if c is not None:
            centers.append(c)
    if not centers:
        return None
    return int(np.median(centers))


# ==========================
# 4) 512 CROP (W ekseninde)
# ==========================
def crop_width_512(vol: np.ndarray, y_center: int, out_w: int = 512) -> np.ndarray:
    W = vol.shape[1]
    half = out_w // 2

    left = y_center - half
    right = left + out_w

    # clamp
    if left < 0:
        left = 0
        right = out_w
    if right > W:
        right = W
        left = W - out_w

    if vol.ndim == 2:
        return vol[:, left:right]
    else:
        return vol[:, left:right, :]


# ==========================
# 5) İSİM ÇIKARMA + KAYDETME
# ==========================
def extract_patient_id(filename: str) -> str:
    base = Path(filename).name
    m = re.match(r"^(\d+)", base)
    if not m:
        raise ValueError(f"Hasta numarası bulunamadı: {filename}")
    return m.group(1)


def make_png_name(patient_id: str, slice_index: int, one_based: bool = True) -> str:
    s = slice_index + 1 if one_based else slice_index
    return f"{patient_id}_slice{s:04d}.png"


def save_slices_as_png(ct_u8: np.ndarray, mask: np.ndarray, out_img_dir: Path, out_lbl_dir: Path,
                      patient_id: str, one_based: bool = True):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    S = ct_u8.shape[2]

    for s in range(S):
        name = make_png_name(patient_id, s, one_based=one_based)

        img2d = ct_u8[:, :, s]  # uint8 (512,512)
        m2d = (mask[:, :, s] > 0).astype(np.uint8) * 255  # uint8 (512,512)

        # PIL ile kaydet
        Image.fromarray(img2d).save(out_img_dir / name)
        Image.fromarray(m2d).save(out_lbl_dir / name)

# ==========================
# 6) READER2 INDEX (EŞLEŞTİRME)
# ==========================
def build_reader2_index(r2_dir: Path):
    mask_map = {}
    for p in r2_dir.glob("*.nii*"):
        if "_mask_2" not in p.name:
            continue
        pid = extract_patient_id(p.name)
        mask_map[pid] = p
    return mask_map


# ==========================
# EK: EKSEN KONTROLÜ (SADECE GÜVENLİK)
# ==========================
def ensure_shape_is_512xYxS(vol: np.ndarray) -> np.ndarray:
    """
    Beklenen: (512, Y, S). Bazen NIfTI eksenleri ters gelebilir.
    - Eğer vol.shape[1] == 512 ve vol.shape[0] != 512 ise, (H,W,S) düzeltmesi için swap yapar.
    """
    if vol.ndim != 3:
        raise ValueError(f"3D volume bekleniyordu, gelen shape={vol.shape}")

    H, W, S = vol.shape
    if H == 512:
        return vol

    # Eğer W=512 ise, H ile W yer değiştirmiş olabilir
    if W == 512:
        return np.transpose(vol, (1, 0, 2))  # (W,H,S) -> (H,W,S)

    # Hiçbiri 512 değilse, burada durup incelemek daha doğru
    raise ValueError(f"Beklenmeyen shape (512 yok): {vol.shape}")


# ==========================
# MAIN
# ==========================
def main():
    if not CT_DIR.exists():
        raise FileNotFoundError(f"CT_DIR bulunamadı: {CT_DIR}")
    if not R2_DIR.exists():
        raise FileNotFoundError(f"R2_DIR bulunamadı: {R2_DIR}")

    mask_map = build_reader2_index(R2_DIR)

    ct_files = sorted(CT_DIR.glob("*.nii*"))
    print(f"CT sayısı: {len(ct_files)}")
    print(f"Reader2 mask sayısı: {len(mask_map)}")
    print(f"Çıktı klasörü: {OUT_BASE}")

    processed = 0
    skipped = 0

    for ct_path in ct_files:
        pid = extract_patient_id(ct_path.name)
        r2_path = mask_map.get(pid)

        if r2_path is None:
            print(f"[SKIP] Mask yok: CT={ct_path.name}")
            skipped += 1
            continue

        # 1) Oku
        ct_vol, _, _ = load_nifti(ct_path)
        r2_vol, _, _ = load_nifti(r2_path)

        # 2) Eksenleri güvenli hale getir (512 x Y x S)
        ct_vol = ensure_shape_is_512xYxS(ct_vol)
        r2_vol = ensure_shape_is_512xYxS(r2_vol)

        # 3) Boyut kontrolü (slice sayısı eşit mi?)
        if ct_vol.shape != r2_vol.shape:
            print(f"[SKIP] Shape uyuşmuyor: {pid} CT={ct_vol.shape} R2={r2_vol.shape}")
            skipped += 1
            continue

        H, W, S = ct_vol.shape
        if W < 512:
            print(f"[SKIP] W<512 çıktı (beklenmiyor): {pid} shape={ct_vol.shape}")
            skipped += 1
            continue

        # 4) Window -> uint8 (makale ile aynı uzay)
        ct_u8 = window_to_uint8(ct_vol, level=WL, width=WW)

        # 5) Median merkez (Reader2 mask'ten)
        y_center = volume_median_center_y(r2_vol, min_pixels=MIN_MASK_PIXELS)
        if y_center is None:
            # Maske çok boş/azsa fallback: ortadan crop
            y_center = W // 2
            print(f"[WARN] {pid}: Mask çok boş; fallback center={y_center}")

        # 6) Crop (CT + mask aynı pencere!)
        ct_u8_c = crop_width_512(ct_u8, y_center=y_center, out_w=512)
        r2_c    = crop_width_512(r2_vol, y_center=y_center, out_w=512)

        # 7) Son kontrol: genişlik tam 512 mi?
        if ct_u8_c.shape[1] != 512 or r2_c.shape[1] != 512:
            print(f"[SKIP] Crop sonrası genişlik 512 değil: {pid} CT={ct_u8_c.shape} R2={r2_c.shape}")
            skipped += 1
            continue

        # 8) Kaydet (aynı isimle image ve label)
        save_slices_as_png(ct_u8_c, r2_c, OUT_IMG, OUT_LBL, patient_id=pid, one_based=ONE_BASED_SLICE)

        processed += 1
        print(f"[OK] {pid}: shape={H}x{W}x{S} center={y_center} -> saved")

    print("===================================")
    print("OUT_BASE =", OUT_BASE.resolve())
    print("OUT_IMG  =", OUT_IMG.resolve())
    print("OUT_LBL  =", OUT_LBL.resolve())
    print(f"İşlenen: {processed}")
    print(f"Atlanan: {skipped}")
    print("Bitti.")


if __name__ == "__main__":
    main()