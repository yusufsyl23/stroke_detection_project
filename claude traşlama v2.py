"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               BEYİN TRAŞLAMA — GÜVENLİ RAY-MARCHING v2.0                  ║
║                                                                              ║
║  Orijinal koddan farklar (3 kritik düzeltme):                                ║
║                                                                              ║
║  DÜZELTME 1 — Derinlik sınırı (MAX_STRIP_DEPTH):                            ║
║    Orijinalde sınır yoktu. Algoritma t_white koşulu sağlandığı sürece        ║
║    görüntünün sonuna kadar gidiyordu. İnce/küçük dilimlerde beynin           ║
║    içinden geçip öbür tarafa çıkıyordu. 10 adet %100-yenme vakasının        ║
║    temel sebebi buydu.                                                        ║
║                                                                              ║
║  DÜZELTME 2 — İnce dilim tespiti (is_thin_slice):                           ║
║    Non-background piksel oranı < %5 olan dilimler "ince/uç" dilimdir.       ║
║    Bu dilimlerde beyin dokusu da yüksek yoğunluklu görünür çünkü            ║
║    doku çok az. Orijinal t_white=229 bu dokuyu da siliyor.                  ║
║    Çözüm: ince dilimlerde t_white = max(245, 0.98×max) → doku korunur.     ║
║                                                                              ║
║  DÜZELTME 3 — Bağımsız yön maskeleri:                                       ║
║    Orijinalde 4 yön tek 'out' dizisini paylaşıyordu. Sol→Sağ bir pikseli   ║
║    silince, Sağ→Sol o pikseli zaten 0 görüp atlıyordu → çift sayım /        ║
║    kaçırma. Çözüm: her yön kendi bağımsız maskesine yazar, sonunda OR.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# >>> KLASÖR YOLLARINI BURADAN DEĞİŞTİR
# ============================================================================
INPUT_DIR  = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\SiyahVeKaymaTemizBrainMaskUyg"
OUTPUT_DIR = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Sadece Traşlama V2 Traşlama"

# ── Güvenlik Parametreleri ───────────────────────────────────────────────────
# Maksimum soyma derinliği (piksel): Her yönde en fazla bu kadar piksel silinir.
# Orijinal kodda bu sınır YOKTU. 512px görüntü için 25 önerilir.
MAX_STRIP_DEPTH = 25

# t_white hesaplama: max(MAX_T_WHITE_FLOOR, SAFETY_FACTOR × img_max)
# Orijinal: max(200, 0.90 × max) → yeni: max(220, 0.92 × max) → daha temkinli
MAX_T_WHITE_FLOOR = 220   # orijinal: 200
SAFETY_FACTOR     = 0.92  # orijinal: 0.90

# İnce dilim eşiği: non-background piksel oranı bu değerin altındaysa
# dilim "ince/uç" sayılır ve çok daha temkinli eşik kullanılır.
THIN_SLICE_THRESHOLD = 0.05  # %5

# ============================================================================

import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# OTOMATİK EŞİK HESAPLAMA
# ─────────────────────────────────────────────────────────────────────────────

def auto_thresholds(img_u8: np.ndarray) -> tuple[int, int, bool, float]:
    """
    Görüntünün kenar piksellerinden arka plan (t_bg) ve beyaz (t_white)
    eşiklerini hesapla.

    Döndürür: (t_bg, t_white, is_thin_slice, non_bg_pct)
    """
    h, w = img_u8.shape
    """
    Görüntünün dört kenarından 5 piksel kalınlığında çerçeve al. Toplam piksel sayısı: 5×512×4 = 10240 piksel. 
    BT'de beyin daima ortada → kenar = hava. P99.9 → kenardaki en parlak %0.1 gürültüyü tolere et, gerisi arka plan. 
    Sonuç tipik olarak 0-5 arası. V1 ile aynı, değişmedi.
    """
    border = np.concatenate([
        img_u8[:5,  :].ravel(),
        img_u8[-5:, :].ravel(),
        img_u8[:,  :5].ravel(),
        img_u8[:, -5:].ravel()
    ])
    t_bg = int(np.percentile(border, 99.9))
    mx   = int(img_u8.max())

    # ── İnce dilim tespiti ───────────────────────────────────────────────────
    # Bundan kastı resimde beyin çok küçük. Yani hem beyaz alan hem de gri(beyin) alan toplamı çok küçük
    # Non-background piksel oranı hesapla
    """
    img_u8 > t_bg → arka plan olmayan piksel sayısı. / img_u8.size → 512×512=262144'e böl → oran. Bu V1'de yoktu. 
    Amacı: dilimin ne kadar "dolu" olduğunu ölç. 
    
    * Orta dilim (beyin gövdesi): %15-25 civarı
    * İnce uç dilim: %1-4 civarı
    * 0049 gibi slice: çok düşük
    """
    non_bg_pct = float((img_u8 > t_bg).sum()) / img_u8.size

    if non_bg_pct < THIN_SLICE_THRESHOLD:   # %5'ten az   beyin çok küçükse
        # İnce/uç dilim: çok yüksek eşik → beyin dokusu silinmez
        # Test: 240 yoğunluklu beyin dokusu, t_white=249 → 240 < 249 → KORUNUR ✅
        t_white   = int(max(245, 0.98 * mx))
        is_thin   = True
    else:
        t_white   = int(max(MAX_T_WHITE_FLOOR, SAFETY_FACTOR * mx))     #220-234 arası
        is_thin   = False

    return t_bg, t_white, is_thin, non_bg_pct


# ─────────────────────────────────────────────────────────────────────────────
# GÜVENLİ RAY-MARCHING
# ─────────────────────────────────────────────────────────────────────────────

def strip_halo(img_u8: np.ndarray) -> tuple:
    """
    4 yönlü güvenli ray-marching tıraşlama.

    Döndürür:
        cleaned   : tıraşlanmış görüntü (np.uint8)
        halo_mask : tıraşlanan bölge maskesi (np.uint8, beyaz=tıraşlandı)
        t_bg      : kullanılan arka plan eşiği
        t_white   : kullanılan beyaz eşiği
        is_thin   : dilim ince/uç olarak tespit edildi mi
        non_bg_pct: non-background piksel oranı (0-1)
        dir_pixels: yön bazlı tıraşlanan piksel sayıları (dict)
    """
    t_bg, t_white, is_thin, non_bg_pct = auto_thresholds(img_u8)
    h, w = img_u8.shape

    # Her yön BAĞIMSIZ maskesine yazar — orijinal img_u8 değişmez
    # Orijinal kodda tek 'out' paylaşılıyordu → yön çakışması problemi
    mask_L = np.zeros((h, w), dtype=np.uint8)  # Sol → Sağ
    mask_R = np.zeros((h, w), dtype=np.uint8)  # Sağ → Sol
    mask_T = np.zeros((h, w), dtype=np.uint8)  # Yukarı → Aşağı
    mask_B = np.zeros((h, w), dtype=np.uint8)  # Aşağı → Yukarı

    # ── Sol → Sağ ────────────────────────────────────────────────────────────
    for y in range(h):
        x = 0
        # 1) Arka planı atla
        while x < w and img_u8[y, x] <= t_bg:
            x += 1
        # 2) Parlak pikselleri sil — MAX_STRIP_DEPTH sınırı ile
        depth = 0
        while x < w and img_u8[y, x] >= t_white and depth < MAX_STRIP_DEPTH:
            mask_L[y, x] = 255
            x += 1
            depth += 1

    # ── Sağ → Sol ────────────────────────────────────────────────────────────
    for y in range(h):
        x = w - 1
        while x >= 0 and img_u8[y, x] <= t_bg:
            x -= 1
        depth = 0
        while x >= 0 and img_u8[y, x] >= t_white and depth < MAX_STRIP_DEPTH:
            mask_R[y, x] = 255
            x -= 1
            depth += 1

    # ── Yukarı → Aşağı ───────────────────────────────────────────────────────
    for x in range(w):
        y = 0
        while y < h and img_u8[y, x] <= t_bg:
            y += 1
        depth = 0
        while y < h and img_u8[y, x] >= t_white and depth < MAX_STRIP_DEPTH:
            mask_T[y, x] = 255
            y += 1
            depth += 1

    # ── Aşağı → Yukarı ───────────────────────────────────────────────────────
    for x in range(w):
        y = h - 1
        while y >= 0 and img_u8[y, x] <= t_bg:
            y -= 1
        depth = 0
        while y >= 0 and img_u8[y, x] >= t_white and depth < MAX_STRIP_DEPTH:
            mask_B[y, x] = 255
            y -= 1
            depth += 1

    # Birleşik maske: 4 yönün birleşimi
    # 4 maskeyi bitwise OR ile birleştir. Bir piksel herhangi bir yönden işaretlenmişse final maskede 255 olur.
    halo_mask = ((mask_L | mask_R | mask_T | mask_B) > 0).astype(np.uint8) * 255

    # Tıraşlanmış görüntü
    # Orijinalin kopyasını al, maskeli pikselleri 0 yap.
    cleaned = img_u8.copy()
    cleaned[halo_mask > 0] = 0

    # Yön bazlı istatistik
    # Her yön kaç piksel işaretledi?
    dir_pixels = {
        'left':   int((mask_L > 0).sum()),
        'right':  int((mask_R > 0).sum()),
        'top':    int((mask_T > 0).sum()),
        'bottom': int((mask_B > 0).sum()),
    }

    return cleaned, halo_mask, t_bg, t_white, is_thin, non_bg_pct, dir_pixels


# ─────────────────────────────────────────────────────────────────────────────
# WORKER — tek dosya işle
# ─────────────────────────────────────────────────────────────────────────────

def process_file(file: Path, clean_dir: Path, mask_dir: Path) -> dict:
    try:
        img = np.array(Image.open(file).convert("L"), dtype=np.uint8)

        cleaned, halo_mask, t_bg, t_white, is_thin, non_bg_pct, dir_px = strip_halo(img)

        # Kaydet
        Image.fromarray(cleaned).save(clean_dir / file.name)
        Image.fromarray(halo_mask).save(mask_dir  / file.name)

        halo_count = int((halo_mask > 0).sum())
        total_px   = img.size

        return {
            "filename":            file.name,
            "height":              img.shape[0],
            "width":               img.shape[1],
            "img_min":             int(img.min()),
            "img_max":             int(img.max()),
            "img_mean":            round(float(img.mean()), 3),
            "img_std":             round(float(img.std()),  3),
            "img_p95":             float(np.percentile(img, 95)),
            "img_p99":             float(np.percentile(img, 99)),
            "t_bg":                t_bg,
            "t_white":             t_white,
            "is_thin_slice":       is_thin,
            "non_bg_pct":          round(non_bg_pct * 100, 4),
            "halo_pixels_removed": halo_count,
            "halo_pct":            round(halo_count / total_px * 100, 4),
            "dir_left_px":         dir_px["left"],
            "dir_right_px":        dir_px["right"],
            "dir_top_px":          dir_px["top"],
            "dir_bottom_px":       dir_px["bottom"],
            "dir_dominant":        max(dir_px, key=dir_px.get) if halo_count > 0 else "none",
            "dir_imbalance":       round(
                max(dir_px.values()) / (sum(dir_px.values()) + 1e-9), 4
            ) if halo_count > 0 else 0.0,
            "error": "",
        }

    except Exception as e:
        return {
            "filename": file.name, "error": str(e),
            "height": 0, "width": 0, "img_min": 0, "img_max": 0,
            "img_mean": 0, "img_std": 0, "img_p95": 0, "img_p99": 0,
            "t_bg": 0, "t_white": 0, "is_thin_slice": False,
            "non_bg_pct": 0, "halo_pixels_removed": 0, "halo_pct": 0,
            "dir_left_px": 0, "dir_right_px": 0, "dir_top_px": 0,
            "dir_bottom_px": 0, "dir_dominant": "error", "dir_imbalance": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ANA PROGRAM
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 65)
    print(" 🧠  BEYİN TRAŞLAMA v2.0 — GÜVENLİ RAY-MARCHING")
    print("=" * 65)
    print(f"  MAX_STRIP_DEPTH    : {MAX_STRIP_DEPTH} px  (orijinalde SINIR YOKTU)")
    print(f"  MAX_T_WHITE_FLOOR  : {MAX_T_WHITE_FLOOR}   (orijinalde: 200)")
    print(f"  SAFETY_FACTOR      : {SAFETY_FACTOR}  (orijinalde: 0.90)")
    print(f"  THIN_SLICE_EŞIĞI   : %{THIN_SLICE_THRESHOLD*100:.0f} non-bg piksel")
    print()

    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    clean_dir  = output_dir / "cleaned_png"
    mask_dir   = output_dir / "halo_mask_png"
    clean_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))
    print(f"  📁 Bulunan PNG    : {len(png_files)}")
    if not png_files:
        print("  ❌ Dosya bulunamadı! INPUT_DIR'i kontrol edin.")
        sys.exit(1)

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"  ⚙️  CPU çekirdeği  : {n_workers} / {mp.cpu_count()}")
    print()

    worker = partial(process_file, clean_dir=clean_dir, mask_dir=mask_dir)

    rows, done = [], 0
    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(worker, png_files, chunksize=4):
            rows.append(result)
            done += 1
            if done % 100 == 0 or done == len(png_files):
                print(f"  ⏳ {done}/{len(png_files)} tamamlandı...")

    df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)

    # Özet
    errors    = df[df["error"] != ""]
    thin_cnt  = df["is_thin_slice"].sum()
    print(f"\n  ✅ Tamamlandı")
    print(f"  Hata          : {len(errors)}")
    print(f"  İnce dilim    : {thin_cnt} adet ({thin_cnt/len(df)*100:.1f}%) → temkinli eşik kullanıldı")
    print(f"  Ort. halo     : {df['halo_pixels_removed'].mean():.1f} px / dilim")
    print(f"  Baskın yön    : {df['dir_dominant'].value_counts().idxmax()}")

    if len(errors):
        print("\n  ⚠️  Hatalı dosyalar:")
        print(errors[["filename", "error"]].to_string(index=False))

    # Excel kaydet
    excel_path = output_dir / "traslama_v2_istatistik.xlsx"
    df.to_excel(excel_path, index=False)

    print(f"\n  💾 Excel   : {excel_path}")
    print(f"  📁 Cleaned : {clean_dir}")
    print(f"  📁 Mask    : {mask_dir}")
    print("=" * 65)