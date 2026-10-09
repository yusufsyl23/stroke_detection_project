from pathlib import Path
import nibabel as nib
import numpy as np

def load_nifti(path: Path):
    """
    NIfTI dosyasını okur ve 3 şeyi döndürür:
    - vol: 3D numpy array (H, W, S) veya (X, Y, Z) gibi
    - affine: görüntünün dünya koordinatlarına dönüşümü
    - spacing: (dx, dy, dz) voxel boyutu (mm)
    """
    # 1) Dosyayı nibabel ile açıyoruz (header + affine + data)
    img = nib.load(str(path))

    # 2) Asıl voxel verisini numpy array'e çeviriyoruz
    # get_fdata() çoğu zaman float döndürür; HU gibi işler için sorun değil.
    vol = img.get_fdata()

    # 3) Affine matris: voxel -> mm dönüşüm bilgisi (konum/orientasyon)
    affine = img.affine

    # 4) Voxel spacing (mm) bilgisi: header'dan gelir.
    # get_zooms() genelde (dx, dy, dz) döndürür.
    # in-plane ve slice thickness burada.
    spacing = img.header.get_zooms()[:3]

    return vol, affine, spacing


def window_to_uint8(vol_hu: np.ndarray, level: float = 40.0, width: float = 120.0) -> np.ndarray:
    """
    HU (veya HU benzeri) CT değerlerini windowlayıp 0-255 uint8'e çevirir.

    Args:
        vol_hu: numpy array (2D slice ya da 3D volume) - HU ölçeğinde
        level: Window Level (WL)
        width: Window Width (WW)

    Returns:
        vol_u8: aynı shape'te uint8 numpy array (0..255)
    """
    # 1) Window sınırlarını hesapla
    low = level - (width / 2.0)    # -20’nin altındaki her şeyi -20 yapar (tamamen siyaha yakın olur)
    high = level + (width / 2.0)   # 100’ün üstündeki her şeyi 100 yapar (tamamen beyaza yakın olur)

    # 2) Clip: aralığın altını low'a, üstünü high'a kırp
    vol_clipped = np.clip(vol_hu, low, high)

    # 3) Normalize: [low, high] -> [0, 1]
    # -20 100 aralığında olan veriyi 0-1 arasına normalize ettik
    vol_norm = (vol_clipped - low) / (high - low)

    # 4) [0,1] -> [0,255]
    vol_255 = vol_norm * 255.0

    # 5) uint8'e çevir (PNG için). Sayıları 0–255 aralığında tam sayı yapar. PNG kaydederken standart olan budur.
    vol_u8 = vol_255.astype(np.uint8)

    return vol_u8

def slice_bbox_center_y(mask2d: np.ndarray, min_pixels: int = 50):
    """
    Tek bir 2D mask slice'ında (H, W) mask'in kapladığı alanın bbox merkezini bulur.
    Sadece Y eksenindeki merkezi (column center) döndürür.

    Args:
        mask2d: 2D numpy array. 0/1 veya 0/255 olabilir.
        min_pixels: çok küçük/boş maskeleri yok saymak için eşik.

    Returns:
        y_center (int) veya None (mask boşsa)
    """
    # 1) Mask'i boolean'a çevir: >0 olan her şey True
    m = (mask2d > 0)

    # 2) Kaç tane True piksel var? (mask boş mu?)
    if m.sum() < min_pixels:
        return None

    # 3) True olan piksellerin koordinatlarını çıkar
    # ys: satır indexleri (0..H-1), xs: sütun indexleri (0..W-1)
    ys, xs = np.where(m)

    # 4) BBox: x_min, x_max (bizim için sütun ekseni = Y yönü = genişlik)
    x_min = int(xs.min())
    x_max = int(xs.max())

    # 5) BBox merkezini hesapla
    y_center = (x_min + x_max) // 2

    return y_center

def volume_median_center_y(mask3d: np.ndarray, min_pixels: int = 50):
    """
    Medyan almamızın sebebi medyanın uç değerlere karşı dayanıklı olmasıdır. Mean 120 gibi uç değerlerden etkilenir

    centers = [248, 252, 251, 249, 250, 253, 120]

    Mean (ortalama): (248+252+251+249+250+253+120)/7≈232

    Medyan = [120, 248, 249, 250, 251, 252, 253] --> ortanca değer 250. Bu gerçek merkeze çok yakın.

    3D mask hacminde (H, W, S) her slice için bbox merkezini bulur,
    sonra median alarak tek bir y_center döndürür.

    Returns:
        y_center (int) veya (hiç slice geçerli değilse) None
    """
    centers = []

    # 1) Slice slice gez
    for s in range(mask3d.shape[2]):
        c = slice_bbox_center_y(mask3d[:, :, s], min_pixels=min_pixels)
        if c is not None:
            centers.append(c)

    # 2) Hiç geçerli slice yoksa
    if len(centers) == 0:
        return None

    # 3) Median: outlier'a dayanıklı, stabil merkez verir
    y_center = int(np.median(centers))
    return y_center

def crop_width_512(vol: np.ndarray, y_center: int, out_w: int = 512) -> np.ndarray:
    """
    2D veya 3D görüntüyü genişlik (W) ekseninde out_w olacak şekilde crop eder.

    vol shape:
      - 2D ise: (H, W)
      - 3D ise: (H, W, S)

    Args:
        vol: CT slice/volume veya mask slice/volume
        y_center: crop'un merkez sütunu (0..W-1 arası)
        out_w: istenen çıktı genişliği (default 512)

    Returns:
        cropped: vol ile aynı boyut sayısında, W ekseni out_w olacak şekilde kırpılmış çıktı
    """
    # 1) Genişliği (W) öğren. Çünkü görüntü (H, W) veya (H, W, S). Crop’u W ekseni boyunca yapacağız.
    W = vol.shape[1]

    # 2) Crop yarı genişliği. y_center ortada kalacak şekilde kesmek için “yarı pencere” lazımdı.
    half = out_w // 2  # 512 -> 256

    # 3) İlk başta "ideal" pencereyi tanımla
    left  = y_center - half  # Başlayacağı yer left biteceği yer right. Onun için 512 ekliyoruz
    right = left + out_w  # right = left + 512

    # 4) Clamp (sınır kontrolü)
    # Eğer sol sınır 0'ın altına taştıysa pencereyi sağa it
    if left < 0:
        left = 0
        right = out_w

    # Eğer sağ sınır W'yi aştıysa pencereyi sola it
    if right > W:
        right = W
        left = W - out_w

    # 5) Şimdi güvenli kırp. Aynı fonksiyon hem: tek slice (2D) ve tüm volume (3D) için çalışsın istiyoruz. 3D’de slice boyutu : ile korunmalı.
    if vol.ndim == 2:
        cropped = vol[:, left:right]
    else:
        cropped = vol[:, left:right, :]

    return cropped

import re
from pathlib import Path

def extract_patient_id(filename: str) -> str:
    """
    Dosya isminden hasta numarasını (başındaki rakam bloğunu) çekip döndürür.
    Böylece hem 11.nii.gz hem 04_mask_2.nii.gz çalışır.
    Örnek:
      - '11.nii.gz' -> '11'
      - '04_mask_2.nii.gz' -> '04'
    """
    base = Path(filename).name  # sadece dosya adı
    m = re.match(r"^(\d+)", base)  # baştan itibaren rakamları yakala
    if not m:
        raise ValueError(f"Hasta numarası bulunamadı: {filename}")
    return m.group(1)  # '04' gibi

def make_png_name(patient_id: str, slice_index: int, one_based: bool = True) -> str:
    """
    one_based=True seçersek slice isimleri 0001’den başlar.
    Örn patient_id='01', slice_index=10 -> '01_slice0010.png'
    one_based=True ise slice numarası 1'den başlar.
    """
    s = slice_index + 1 if one_based else slice_index
    return f"{patient_id}_slice{s:04d}.png"  # s:04d → sayıyı 4 haneli yazar (10 → 0010)

import os
import numpy as np
import cv2
from pathlib import Path

def save_slices_as_png(ct_u8: np.ndarray, mask: np.ndarray, out_img_dir: Path, out_lbl_dir: Path,
                      patient_id: str, one_based: bool = True):
    """
    ct_u8: (H, W, S) uint8
    mask:  (H, W, S) 0/1 veya 0/255 (biz kaydederken 0/255 yapacağız)
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    S = ct_u8.shape[2]

    for s in range(S):
        name = make_png_name(patient_id, s, one_based=one_based)

        # CT slice (uint8) kaydet
        img2d = ct_u8[:, :, s]
        cv2.imwrite(str(out_img_dir / name), img2d)

        # Mask slice kaydet: 0/255 yapalım (binary png standardı gibi)
        m2d = (mask[:, :, s] > 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_lbl_dir / name), m2d)

from pathlib import Path

def build_reader2_index(r2_dir: Path):
    """
    Reader2 klasöründeki *_mask_2.nii.gz dosyalarını patient_id'ye göre indexler.
    Returns:
        dict: { "11": Path(.../11_mask_2.nii.gz), ... }
    """
    mask_map = {}
    for p in r2_dir.glob("*.nii*"):  # nii veya nii.gz
        name = p.name
        if "_mask_2" not in name:
            continue
        pid = extract_patient_id(name)  # baştaki sayı bloğu
        mask_map[pid] = p
    return mask_map

def find_matching_mask(mask_map: dict, patient_id: str):
    """
    CT'den gelen patient_id için mask dosyasını döndürür.
    Yoksa None döndürür.
    """
    return mask_map.get(patient_id, None)

# https://chatgpt.com/share/699ece8e-750c-8003-8c1b-8c94dc273d4a