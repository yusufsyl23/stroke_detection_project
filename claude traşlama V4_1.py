# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TRAŞLAMA ALGORİTMASI - SÜRÜM 4.1 (traslamaV4.py)                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  V4.0 -> V4.1 DEĞİŞİKLİKLERİ (veri bazlı kalibrasyon):                     ║
║                                                                              ║
║  DÜZELTME 1 - PEAK_SKULL_MIN_POS: 140 -> 100                                ║
║    V4.0da eşik 140 idi. Ancak kafatası kalınlık analizi                    ║
║    SiyahVeKaymaTemizBrainMaskUyg (tıraşlanmamış, beyin maskeli)             ║
║    üzerinde yeniden çalıştırıldı. Gerçek skull_peak dağılımı:               ║
║    Min=89, P5=106, P10=108, P25=123, P50=156.                               ║
║    Eski eşik 140 -> 435 dilimi (%39.1) gereksiz fallback düşürüyordu.     ║
║    Bu dilimlerde fallback, gerçek t_white ort. 143 HU yüksek üretiyordu ║
║    = kafatası piksellerini traşlamıyordu.                                    ║
║                                                                              ║
║  DÜZELTME 2 - FG_TO_MAX_DEPTH_TABLE tamamen yeniden kalibre edildi          ║
║    Önceki tablo V3-traşlanmış görüntülerden türetilmişti (kafatası           ║
║    kısmen silinmiş = gerçek kalınlık ölçülemiyordu).                        ║
║    Yeni tablo orijinal görüntülerden (1253 dilim):                          ║
║      very_thin (< 2%)    : skull_p90 = 34px  -> MAX_DEPTH = 40px            ║
║      thin low  (2-5%)    : skull_p90 = 72px  -> MAX_DEPTH = 83px            ║
║      thin high (5-8%)    : skull_p90 = 90px  -> MAX_DEPTH = 104px           ║
║      normal low (8-15%)  : skull_p90 = 113px -> MAX_DEPTH = 130px           ║
║      normal mid (15-25%) : skull_p90 = 139px -> MAX_DEPTH = 160px           ║
║      normal big (25-36%) : skull_p90 = 168px -> MAX_DEPTH = 193px           ║
║                                                                              ║
║  KORUNAN KRİTERLER (veriyle doğrulandı, değiştirilmedi):                   ║
║    PEAK_MIN_SEPARATION_HU = 40: başarılı tespitlerde min sep=42, DOĞRU     ║
║    PEAK_VALLEY_MAX_RATIO  = 0.60: ratio>0.60 dilimlerinde t_white=65-92,   ║
║      bu değerler beyin dokusu eşiğine çok yakın, fallback ŞART             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  BU SCRIPT TEK SEFERDE ÜÇ ÇIKTI ÜRETIR:                                    ║
║                                                                              ║
║  1. TRAŞLANMIŞ GÖRÜNTÜLER  -> OUTPUT_DIR/                                    ║
║  2. HALO MASKELERI          -> OUTPUT_DIR/halo_mask_png/                     ║
║  3. İSTATİSTİK EXCEL        -> OUTPUT_DIR/v4_istatistik.xlsx                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
#
# KULLANIM:
#     python traslamaV4.py
#
# GEREKSINIMLER:
#     pip install numpy pillow scipy pandas openpyxl

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import binary_erosion
import warnings
warnings.filterwarnings('ignore')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        AYARLAR - BURADAN DEĞİŞTİR                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Orijinal (traşlanmamış) CT görüntüleri - beyin maskesi uygulanmış
INPUT_DIR  = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\yedek\SiyahVeKaymaTemizBrainMaskUyg"
# Traşlanmış görüntülerin kaydedileceği ana klasör
# -> Halo mask alt klasörü ve Excel de buraya kaydedilir
OUTPUT_DIR = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V4_1"

# GT (uzman) mask klasörü - istatistik exceli için beyin bölgesi yoğunlukları
# Eğer GT maskelerin yoksa veya istemiyorsan None yap:  GT_MASK_DIR = None
GT_MASK_DIR = None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     TRAŞLAMA PARAMETRELERİ VE AÇIKLAMALARI                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Dilim boyutu sınırları ─────────────────────────────────────────────────────
VERY_THIN_MAX_FG_PCT = 2.0   # % - çok ince dilim
THIN_MAX_FG_PCT      = 8.0   # % - ince dilim

# ── Dinamik MAX_DEPTH tablosu ──────────────────────────────────────────────────
# Kaynak: kafatasi_kalinlik_istatistik.xlsx (SiyahVeKaymaTemizBrainMaskUyg,
#         traşlama uygulanmamış, beyin maskeli görüntüler - 1253 dilim)
#         skull_thickness_p90 değerleri + %15 güvenlik marjı
#
# Önceki tablo V3-traşlanmış görüntülerden türetilmişti (kafatası kısmen silinmiş).
# Yeni tablo orijinal veri setinden: gerçek kafatası kalınlığı daha yüksek çıktı.
#
#  slice_class    fg_pct aralığı  skull_p90  -> MAX_DEPTH (%15 marj)
#  very_thin      < 2%            34.0 px    -> 40 px
#  thin (düşük)   2-5%            72.2 px    -> 83 px
#  thin (yüksek)  5-8%            90.4 px    -> 104 px
#  normal (düşük) 8-15%          112.5 px    -> 130 px
#  normal (orta)  15-25%         139.4 px    -> 160 px
#  normal (büyük) 25-36%         168.0 px    -> 193 px
#  çok büyük      >= 36%           -          -> 200 px  (güvenlik tavanı)
FG_TO_MAX_DEPTH_TABLE = [
    ( 2.0,  40),   # very_thin:   skull_p90=34.0px  -> 40px
    ( 5.0,  83),   # thin low:    skull_p90=72.2px  -> 83px
    ( 8.0, 104),   # thin high:   skull_p90=90.4px  -> 104px
    (15.0, 130),   # normal low:  skull_p90=112.5px -> 130px
    (25.0, 160),   # normal mid:  skull_p90=139.4px -> 160px
    (35.0, 193),   # normal big:  skull_p90=168.0px -> 193px
    (100.0,200),   # çok büyük (nadir): güvenlik tavanı
]

# ── t_white peak tespiti güvenilirlik kriterleri ────────────────────────────────
# Kaynak: kafatasi_kalinlik_istatistik.xlsx - SiyahVeKaymaTemizBrainMaskUyg
#         (traşlama uygulanmamış, beyin maskeli görüntüler, n=1253)
#
# KRİTER 1 - peak_separation >= 40 HU:
#   Başarılı tespitlerde MIN separation = 42 HU -> eşik 40 DOĞRU, değiştirilmedi.
#
# KRİTER 2 - skull_peak_pos >= 100:
#   Gerçek skull_peak dağılımı: Min=89, P5=106, P10=108, P25=123, P50=156
#   Önceki eşik 140 idi -> başarılı tespitlerin %39.1ini (435 dilim) gereksiz
#   yere fallback düşürüyordu. skull 100-140 arası dilimlerde fallback,
#   gerçek t_white ortalama 143 HU daha yüksek üretiyor = kafatası kalıyor.
#   Yeni eşik: 100 -> sadece gerçekten anlamsız (< 100) pikler reddedilir.
#
# KRİTER 3 - valley_depth_ratio <= 0.60:
#   valley_ratio > 0.60 olan dilimlerde t_white_new = 65-92 arası (çok düşük).
#   Bu dilimler için fallback 165 HU daha yüksek -> DOĞRU karar, KORUNDU.
#   valley_ratio > 0.60 -> vadi belirsiz -> t_white güvenilmez -> fallback şart.
PEAK_MIN_SEPARATION_HU  = 40    # iki pik arası min mesafe (HU) - değişmedi
PEAK_SKULL_MIN_POS      = 100   # 140 -> 100: gerçek min=89, P5=106 (bkz. analiz)
PEAK_VALLEY_MAX_RATIO   = 0.60  # vadi derinlik oranı - değişmedi, kritik kriter
PEAK_SMOOTHING_WINDOW   = 11    # Savitzky-Golay pencere boyutu

# ── Fallback t_white ───────────────────────────────────────────────────────────
# V2 formülü: max(220, 0.92 x img_max) - en güvenilir sabit yöntem.
FALLBACK_T_WHITE_FACTOR = 0.92
FALLBACK_T_WHITE_MIN    = 220

# ── Işın yürüyüşü ──────────────────────────────────────────────────────────────
N_DIRECTIONS  = 8   # 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

# ── Güvenlik sınırları ─────────────────────────────────────────────────────────
MIN_MAX_DEPTH  = 8   # asla 8px altına inme
T_WHITE_MARGIN = 5   # t_white = min(t_white, img_max - T_WHITE_MARGIN)

# ── Feedback loop ──────────────────────────────────────────────────────────────
# Yeterli traşlama yoksa MAX_DEPTH artır (t_white dokunma - V3 fark)
FEEDBACK_TARGET_SHAVE_PCT = 0.5   # fg en az bu %si traşlanmalı
FEEDBACK_DEPTH_INCREMENT  = 15    # her turda eklenen derinlik (px)
FEEDBACK_MAX_ITERATIONS   = 4     # maksimum tekrar sayısı
FEEDBACK_MAX_DEPTH_CAP    = 215   # tavan: yeni tablonun üst sınırı (200) + 1 feedback


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         TRAŞLAMA FONKSİYONLARI                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_t_bg(img: np.ndarray, border_width: int = 5) -> int:
    """
    Görüntü kenarlarından arka plan eşiği hesaplar.
    Kenar piksellerinin %99.9 yüzdesi kullanılır.
    cv2 yerine PIL kullandığımız için bu fonksiyon her zaman düzgün çalışır.
    """
    border = np.concatenate([
        img[:border_width, :].ravel(),
        img[-border_width:, :].ravel(),
        img[:, :border_width].ravel(),
        img[:, -border_width:].ravel(),
    ])
    return int(np.percentile(border, 99.9))


def classify_slice(fg_pct: float) -> str:
    """fg_pct degerine gore very_thin / thin / normal siniflandirmasi."""
    if fg_pct <= VERY_THIN_MAX_FG_PCT:
        return "very_thin"
    elif fg_pct <= THIN_MAX_FG_PCT:
        return "thin"
    return "normal"


def get_dynamic_max_depth(fg_pct: float) -> int:
    # fg_pct degerine gore MAX_DEPTH dondurur.
    # FG_TO_MAX_DEPTH_TABLE tablosundan turetilmistir.
    # V3 sabit 20px kullaniyordu. Bu fonksiyon 40-193px arasi dinamik deger uretir.
    for fg_upper, depth in FG_TO_MAX_DEPTH_TABLE:
        if fg_pct <= fg_upper:
            return max(depth, MIN_MAX_DEPTH)
    return FEEDBACK_MAX_DEPTH_CAP


def _fallback_t_white(img_max: int) -> int:
    """
    Peak tespiti güvenilmez olduğunda kullanılan güvenli eşik.
    V2 formulu: max(220, 0.92 x img_max).
    """
    return max(int(FALLBACK_T_WHITE_FACTOR * img_max), FALLBACK_T_WHITE_MIN)


def detect_t_white(img: np.ndarray, fg_mask: np.ndarray,
                   img_max: int) -> tuple:
    """
    Histogram analizi ile kafatası eşiği (t_white) tespit eder.

    Yöntem:
      1. Ön plan piksellerinin histogramını al
      2. Savitzky-Golay ile düzelt
      3. find_peaks ile tüm pikleri bul
      4. En büyük 2 piki seç (küçük=beyin, büyük=kafatası)
      5. Aralarındaki minimum = t_white

    3 guvenilirlik kontrolu gecilmezse fallback devreye girer.

    Döndürür: (t_white: int, method: str, debug: dict)
    """
    debug = {}
    fg_pixels = img[fg_mask].astype(np.int32)

    if len(fg_pixels) < 50:
        return _fallback_t_white(img_max), "fallback_too_few_pixels", debug

    hist, _ = np.histogram(fg_pixels, bins=256, range=(0, 255))

    # Savitzky-Golay: gürültüyü temizler, pik konumlarını korur
    if len(hist) >= PEAK_SMOOTHING_WINDOW:
        hist_s = savgol_filter(hist.astype(float),
                               window_length=PEAK_SMOOTHING_WINDOW,
                               polyorder=2)
        hist_s = np.clip(hist_s, 0, None)
    else:
        hist_s = hist.astype(float)

    # Tüm yerel maksimumları bul
    # distance=15: iki pik arası min 15 birim
    # prominence=10: çevresine göre en az 10 birim çıkıntılı
    peaks, _ = find_peaks(hist_s, distance=15, prominence=10)
    debug["n_peaks"] = len(peaks)

    if len(peaks) < 2:
        return _fallback_t_white(img_max), "fallback_less_than_2_peaks", debug

    # En büyük 2 piki konuma göre sırala
    top2 = np.sort(peaks[np.argsort(hist_s[peaks])[-2:]])
    brain_peak = int(top2[0])   # düşük yoğunluk = beyin
    skull_peak = int(top2[1])   # yüksek yoğunluk = kafatası
    separation = skull_peak - brain_peak

    debug.update({"brain_peak": brain_peak, "skull_peak": skull_peak,
                  "separation": separation})

    # Kontrol 1: Pik ayrımı
    if separation < PEAK_MIN_SEPARATION_HU:
        debug["fail"] = f"sep={separation}"
        return _fallback_t_white(img_max), "fallback_peaks_too_close", debug

    # Kontrol 2: Kafatası piki konumu
    if skull_peak < PEAK_SKULL_MIN_POS:
        debug["fail"] = f"skull={skull_peak}"
        return _fallback_t_white(img_max), "fallback_skull_peak_too_low", debug

    # İki pik arasındaki minimum = vadi noktası = t_white
    segment      = hist_s[brain_peak: skull_peak + 1]
    valley_pos   = brain_peak + int(np.argmin(segment))
    brain_h      = hist_s[brain_peak]
    valley_h     = hist_s[valley_pos]
    valley_ratio = (valley_h / brain_h) if brain_h > 0 else 1.0

    debug.update({"valley_pos": valley_pos,
                  "valley_ratio": round(valley_ratio, 3)})

    # Kontrol 3: Vadi derinliği
    if valley_ratio > PEAK_VALLEY_MAX_RATIO:
        debug["fail"] = f"ratio={valley_ratio:.2f}"
        return _fallback_t_white(img_max), "fallback_shallow_valley", debug

    # Güvenlik kırpması - V3 thin bug önlemi
    t_white = int(valley_pos)
    t_white = min(t_white, img_max - T_WHITE_MARGIN)
    t_white = max(t_white, 120)   # çok düşük eşik beyin yer

    return t_white, "two_peak_valley", debug


def find_fg_boundary(fg_mask: np.ndarray) -> list:
    """
    Ön plan maskesinin sınır piksellerini bulur (erozyon yöntemi).
    Işınlar buradan başlatılır.
    """
    eroded   = binary_erosion(fg_mask)
    boundary = fg_mask & ~eroded
    ys, xs   = np.where(boundary)
    return list(zip(ys.tolist(), xs.tolist()))


def ray_march(img: np.ndarray, start_y: int, start_x: int,
              dy: float, dx: float, t_white: int,
              max_depth: int, fg_mask: np.ndarray) -> np.ndarray:
    """
    Tek yönde ışın yürütür ve traşlama maskesini döndürür.

    Adım adım:
      - Görüntü/ön plan sınırı aşıldı    dur
      - max_depth aşıldı                 dur
      - piksel >= t_white                kafatası, traşla ve devam
      - piksel < t_white                 beyin dokusu, dur

    Her yon kendi maskesini dondurur (V1 cakisma hatasi onlendi).
    """
    h, w   = img.shape
    mask   = np.zeros((h, w), dtype=bool)
    depth  = 0
    cy, cx = float(start_y), float(start_x)

    while True:
        cy += dy;  cx += dx;  depth += 1
        iy, ix = int(round(cy)), int(round(cx))

        if iy < 0 or iy >= h or ix < 0 or ix >= w:  break
        if not fg_mask[iy, ix]:                       break
        if depth > max_depth:                         break

        if img[iy, ix] >= t_white:
            mask[iy, ix] = True   # kafatası: traşla, devam et
        else:
            break                 # beyin: dur

    return mask


def shave_slice(img: np.ndarray, fg_mask: np.ndarray,
                t_white: int, max_depth: int) -> np.ndarray:
    """
    Tüm yönlerde ışın yürütür, OR ile birleştirir.

    N_DIRECTIONS kadar eşit açılı yön (varsayılan 8).
    Sınır piksellerinden başlar, içe doğru giden yönleri işler.
    """
    h, w = img.shape
    final = np.zeros((h, w), dtype=bool)

    # Yön vektörleri
    directions = [
        (np.sin(2 * np.pi * i / N_DIRECTIONS),
         np.cos(2 * np.pi * i / N_DIRECTIONS))
        for i in range(N_DIRECTIONS)
    ]

    boundary = find_fg_boundary(fg_mask)

    for (sy, sx) in boundary:
        for (dy, dx) in directions:
            ny, nx = int(round(sy + dy)), int(round(sx + dx))
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if not fg_mask[ny, nx]:
                continue   # dışa giden yön
            final |= ray_march(img, sy, sx, dy, dx,
                                t_white, max_depth, fg_mask)
    return final


def apply_shave_with_feedback(img: np.ndarray, fg_mask: np.ndarray,
                               t_white: int, initial_max_depth: int,
                               fg_px: int) -> tuple:
    """
    Traşlama + feedback loop.

    Hedef traşlama oranı karşılanmazsa MAX_DEPTH artırılır.
    t_white degerine kesinlikle dokunulmaz (V3 farkli: t_white artiriyordu).

    Döndürür: (shave_mask, final_max_depth, n_iterations)
    """
    max_depth  = initial_max_depth
    shave_mask = None

    for iteration in range(FEEDBACK_MAX_ITERATIONS + 1):
        shave_mask  = shave_slice(img, fg_mask, t_white, max_depth)
        shave_pct   = shave_mask.sum() / max(fg_px, 1) * 100.0

        if shave_pct >= FEEDBACK_TARGET_SHAVE_PCT:
            break  # yeterli

        if iteration < FEEDBACK_MAX_ITERATIONS:
            max_depth = min(max_depth + FEEDBACK_DEPTH_INCREMENT,
                            FEEDBACK_MAX_DEPTH_CAP)
            # NOT: t_white değişmiyor

    return shave_mask, max_depth, iteration


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       İSTATİSTİK YARDIMCI FONKSİYONLARI                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def px_stats(pixels: np.ndarray, prefix: str) -> dict:
    """
    Bir piksel dizisinin tanımlayıcı istatistiklerini döndürür.

    Kullanım: halo piksellerinin yoğunluk dağılımı, GT beyin bölgesi yoğunlukları.

    Hesaplanan değerler: count, mean, std, min, p5, p10, p25, p50, p75, p90, p95, max
    prefix ornegi: "halo" -> "halo_mean", "halo_p95", vb.
    """
    if len(pixels) == 0:
        keys = ['count', 'mean', 'std', 'min',
                'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'max']
        return {f'{prefix}_{k}': np.nan for k in keys}

    return {
        f'{prefix}_count': len(pixels),
        f'{prefix}_mean' : round(float(np.mean(pixels)),   2),
        f'{prefix}_std'  : round(float(np.std(pixels)),    2),
        f'{prefix}_min'  : int(np.min(pixels)),
        f'{prefix}_p5'   : float(np.percentile(pixels,  5)),
        f'{prefix}_p10'  : float(np.percentile(pixels, 10)),
        f'{prefix}_p25'  : float(np.percentile(pixels, 25)),
        f'{prefix}_p50'  : float(np.percentile(pixels, 50)),
        f'{prefix}_p75'  : float(np.percentile(pixels, 75)),
        f'{prefix}_p90'  : float(np.percentile(pixels, 90)),
        f'{prefix}_p95'  : float(np.percentile(pixels, 95)),
        f'{prefix}_max'  : int(np.max(pixels)),
    }


def build_excel(all_rows: list, output_excel: Path) -> None:
    """
    Dilim bazlı istatistikleri alır, 3 sheetlik Excel dosyası üretir.

    Sheet 1 - Tüm Dilimler:
        Her satır = bir dilim. Traşlama parametreleri + yoğunluk istatistikleri.

    Sheet 2 - Hasta Özeti:
        patient_id bazında dilim sayısı, ortalama metrikler.

    Sheet 3 - Dilim Sınıfı Özeti:
        very_thin / thin / normal sınıflarında özet istatistikler.
    """
    df = pd.DataFrame(all_rows).sort_values('filename').reset_index(drop=True)

    # patient_id: dosya adındaki ilk sayısal kısım (örn. "02_slice0067" -> "02")
    df['patient_id'] = df['filename'].str.extract(r'^(\d+)_').astype(str)

    with pd.ExcelWriter(str(output_excel), engine='openpyxl') as writer:

        # ── Sheet 1: Tüm Dilimler ────────────────────────────────────────────
        df.to_excel(writer, sheet_name='Tüm Dilimler', index=False)

        # ── Sheet 2: Hasta Özeti ─────────────────────────────────────────────
        # Her hasta için dilim sayısı ve temel metriklerin ortalaması
        agg_cols = {
            'filename': 'count',
            'fg_pct': 'mean',
            'shaved_pct_of_fg': 'mean',
            'max_depth_final': 'mean',
            'feedback_iterations': 'sum',
            'halo_px': 'sum',
        }
        # Sadece var olan sütunları kullan
        agg_valid = {k: v for k, v in agg_cols.items() if k in df.columns}
        hasta_df  = df.groupby('patient_id', as_index=False).agg(agg_valid)
        hasta_df.rename(columns={'filename': 'n_slices'}, inplace=True)
        hasta_df.to_excel(writer, sheet_name='Hasta Özeti', index=False)

        # ── Sheet 3: Dilim Sınıfı Özeti ──────────────────────────────────────
        if 'slice_class' in df.columns:
            sinif_cols = {
                'filename': 'count',
                'fg_pct': 'mean',
                'shaved_pct_of_fg': 'mean',
                'max_depth_final': 'mean',
                'halo_px': 'mean',
            }
            sinif_valid = {k: v for k, v in sinif_cols.items() if k in df.columns}
            sinif_df    = df.groupby('slice_class', as_index=False).agg(sinif_valid)
            sinif_df.rename(columns={'filename': 'n_slices'}, inplace=True)
            sinif_df.to_excel(writer, sheet_name='Dilim Sınıfı Özeti', index=False)

    print(f"\n  Excel kaydedildi: {output_excel.name}")
    print(f"  ({len(df)} satır, 3 sheet: Tüm Dilimler / Hasta Özeti / Dilim Sınıfı Özeti)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         ANA İŞLEM FONKSİYONU                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def process_single_image(img_path: Path,
                          halo_dir: Path,
                          gt_dir_path) -> tuple:
    """
    Tek bir PNG dosyasını işler. Üç sonuç döner:

      output_img  : Traşlanmış görüntü (kafatası=0, beyin korunmuş)
      halo_mask   : Binary mask (255=traşlanan piksel, 0=dokunulmayan)
      log         : Bu dilim için tüm istatistik ve parametre bilgileri

    İşlem sırası:
      1. Görüntüyü oku (PIL - Türkçe/boşluklu yollar için güvenli)
      2. t_bg -> fg_mask -> fg_pct -> slice_class -> max_depth
      3. t_white tespit et (histogram veya fallback)
      4. t_white güvenlik kırpması (thin bug önlemi)
      5. Traşlama + feedback loop
      6. Halo mask üret ve kaydet
      7. İstatistik hesapla (halo yoğunlukları + GT beyin yoğunlukları + gap)
    """
    log = {'filename': img_path.name}

    try:
        # ── 1. Görüntüyü oku ────────────────────────────────────────────────
        # PIL: cv2 Türkçe karakter sorununu önler
        pil_img  = Image.open(img_path).convert("L")
        img      = np.array(pil_img, dtype=np.uint8)
        h, w     = img.shape
        total_px = h * w
        img_max  = int(img.max())
        img_min  = int(img.min())

        log.update({'img_max': img_max, 'img_min': img_min,
                    'total_px': total_px})

        # ── 2. Ön plan maskesi ve dilim sınıfı ──────────────────────────────
        t_bg    = get_t_bg(img)
        fg_mask = img > t_bg
        fg_px   = int(fg_mask.sum())
        fg_pct  = fg_px / total_px * 100.0

        slice_class = classify_slice(fg_pct)
        max_depth   = get_dynamic_max_depth(fg_pct)

        log.update({'t_bg': t_bg, 'fg_px': fg_px,
                    'fg_pct': round(fg_pct, 4),
                    'slice_class': slice_class,
                    'max_depth_initial': max_depth})

        # Boş dilim: doğrudan orijinali döndür, halo mask sıfır olsun
        if fg_px == 0:
            log['skip_reason'] = 'no_foreground'
            halo_arr = np.zeros((h, w), dtype=np.uint8)
            _save_halo(halo_arr, halo_dir / img_path.name)
            return img.copy(), halo_arr, log

        # ── 3. t_white tespiti ──────────────────────────────────────────────
        t_white, t_method, t_debug = detect_t_white(img, fg_mask, img_max)

        # ── 4. t_white güvenlik kırpması ────────────────────────────────────
        # V3 thin bug: img_max < t_white -> traşlama=0 durumunu önler
        t_white = min(t_white, img_max - T_WHITE_MARGIN)

        log.update({
            't_white': t_white,
            't_white_method': t_method,
            **{f'peak_{k}': v for k, v in t_debug.items()}
        })

        # ── 5. Traşlama + feedback ───────────────────────────────────────────
        shave_mask, final_depth, n_iters = apply_shave_with_feedback(
            img=img, fg_mask=fg_mask,
            t_white=t_white, initial_max_depth=max_depth,
            fg_px=fg_px,
        )

        shaved_px    = int(shave_mask.sum())
        shaved_pct   = shaved_px / max(fg_px, 1) * 100.0

        log.update({
            'max_depth_final': final_depth,
            'feedback_iterations': n_iters,
            'shaved_px': shaved_px,
            'shaved_pct_of_fg': round(shaved_pct, 3),
        })

        # ── 6. Traşlanmış görüntü ────────────────────────────────────────────
        output_img = img.copy()
        output_img[shave_mask] = 0

        # ── 7. Halo mask üret ───────────────────────────────────────────────
        # Traşlanan piksel = orijinalde var (>0) VE V4 sıfır olmuş
        # "Sadece v4==0" demiyoruz çünkü orijinal arka plan da zaten 0dı.
        halo_mask_bool = (img > 0) & (output_img == 0)
        halo_uint8     = halo_mask_bool.astype(np.uint8) * 255

        halo_px  = int(halo_mask_bool.sum())
        halo_pct = halo_px / total_px * 100.0

        log.update({'halo_px': halo_px,
                    'halo_pct': round(halo_pct, 4)})

        _save_halo(halo_uint8, halo_dir / img_path.name)

        # ── 8. Halo yoğunluk istatistikleri ─────────────────────────────────
        # Traşlanan bölgedeki orijinal piksel yoğunlukları
        # -> "kafatası dokusu mu yoksa beyin mi traşlandı?" sorusunun cevabı
        # Beklenti: halo_mean > 180 (kafatası yoğun, beyin ~80-150)
        halo_pixels = img[halo_mask_bool].astype(np.uint8)
        log.update(px_stats(halo_pixels, 'halo'))

        # ── 9. GT beyin istatistikleri (varsa) ──────────────────────────────
        # GT maskesi ile örtüşen CT piksellerinin yoğunluğu
        # gap = skull_p25 - brain_p95 -> negatif ise beyin ve kafatası örtüşüyor
        if gt_dir_path is not None:
            gt_file = Path(gt_dir_path) / img_path.name
            if gt_file.exists():
                gt_bin  = np.array(
                    Image.open(gt_file).convert("L"), dtype=np.uint8) > 127
                brain_px = img[gt_bin].astype(np.uint8)
                log.update(px_stats(brain_px, 'brain'))
                log['brain_mask_px']  = int(gt_bin.sum())
                log['brain_mask_pct'] = round(gt_bin.sum() / total_px * 100, 4)

                # Gap analizi: kafatası en düşük yoğunluk - beyin en yüksek yoğunluk
                # Pozitif gap -> net ayrım var (iyi)
                # Negatif gap -> örtüşme var (eşik belirsiz)
                skull_p25  = log.get('halo_p25', np.nan)
                brain_p95  = log.get('brain_p95', np.nan)
                skull_p5   = log.get('halo_p5', np.nan)

                if not (np.isnan(skull_p25) or np.isnan(brain_p95)):
                    gap = skull_p25 - brain_p95
                    log['gap']              = round(gap, 2)
                    log['gap_negative']     = bool(gap < 0)

                    # Alternatif t_white tahminleri (V5 için referans)
                    log['t_white_midpoint'] = round((skull_p25 + brain_p95) / 2, 1)
                if not np.isnan(skull_p5):
                    log['t_white_skull_p5m10'] = round(skull_p5 - 10, 1)
            else:
                log.update(px_stats(np.array([], dtype=np.uint8), 'brain'))
                log['brain_mask_px']  = np.nan
                log['brain_mask_pct'] = np.nan
                log['gap']            = np.nan
                log['gap_negative']   = np.nan

        return output_img, halo_uint8, log

    except Exception as e:
        log['error'] = str(e)
        # Hata durumunda orijinali döndür, halo sıfır
        try:
            img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        except:
            img = np.zeros((512, 512), dtype=np.uint8)
        halo_empty = np.zeros_like(img)
        return img, halo_empty, log


def _save_halo(halo_arr: np.ndarray, out_path: Path) -> None:
    # Binary halo mask PNG kaydet.
    Image.fromarray(halo_arr).save(str(out_path))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              ANA DÖNGÜ                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    # ANA ISLEV: INPUT_DIR icindeki tum PNGleri isler.
    # 3 cikti uretir: traslanmis goruntuler + halo maskeler + Excel.
    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    halo_path   = output_path / "halo_mask_png"
    excel_path  = output_path / "v4_istatistik.xlsx"

    # Klasörleri oluştur
    output_path.mkdir(parents=True, exist_ok=True)
    halo_path.mkdir(parents=True, exist_ok=True)

    # Giriş dosyalarını listele
    png_files = sorted(input_path.glob("*.png"))

    if not png_files:
        print(f"HATA: {INPUT_DIR} içinde PNG bulunamadı!")
        return

    # GT klasörü kontrolü
    gt_dir = GT_MASK_DIR if (GT_MASK_DIR and Path(GT_MASK_DIR).exists()) else None
    if GT_MASK_DIR and not gt_dir:
        print(f"  UYARI: GT klasörü bulunamadı, gap analizi atlanacak.\n  {GT_MASK_DIR}")

    print(f"\n{'='*72}")
    print(f"  TRAŞLAMA V4  |  {len(png_files)} dosya")
    print(f"  Giriş          : {INPUT_DIR}")
    print(f"  Traşlanmış çıktı: {OUTPUT_DIR}")
    print(f"  Halo mask      : {halo_path}")
    print(f"  Excel istatistik: {excel_path.name}")
    print(f"  GT mask        : {gt_dir or '(kullanılmıyor)'}")
    print(f"{'='*72}\n")

    all_logs = []
    n_ok = n_err = 0

    for i, img_path in enumerate(png_files, 1):

        # İlerleme bildirimi
        if i % 50 == 0 or i == 1 or i == len(png_files):
            print(f"  [{i:4d}/{len(png_files)}] İşleniyor...")

        output_img, halo_arr, log = process_single_image(
            img_path, halo_path, gt_dir
        )

        if 'error' in log:
            n_err += 1
            print(f"  !! HATA: {img_path.name} - {log['error']}")
        else:
            # Traşlanmış görüntüyü kaydet
            Image.fromarray(output_img).save(str(output_path / img_path.name))
            n_ok += 1

            # Dikkat çekici durumları ekrana yaz
            iters  = log.get('feedback_iterations', 0)
            method = log.get('t_white_method', '')
            if i <= 3 or iters > 0 or 'fallback' in str(method):
                print(f"    {img_path.name}: "
                      f"{log.get('slice_class','?'):9s} | "
                      f"t_white={log.get('t_white','?'):3} "
                      f"({str(method)[:18]:18s}) | "
                      f"depth={log.get('max_depth_final','?'):3}px | "
                      f"traş={log.get('shaved_pct_of_fg',0):5.1f}% | "
                      f"halo={log.get('halo_px',0):5}px"
                      + (f" [FBx{iters}]" if iters > 0 else ""))

        all_logs.append(log)

    # ── Excel istatistik ────────────────────────────────────────────────────
    print(f"\n  Excel hazırlanıyor...")
    try:
        build_excel(all_logs, excel_path)
    except Exception as e:
        print(f"  !! Excel hatası: {e}")
        # Yedek: CSV kaydet
        csv_path = output_path / "v4_istatistik.csv"
        pd.DataFrame(all_logs).to_csv(str(csv_path), index=False,
                                       encoding='utf-8-sig')
        print(f"  Yedek CSV kaydedildi: {csv_path.name}")

    # ── Özet ────────────────────────────────────────────────────────────────
    valid_logs = [r for r in all_logs if 'error' not in r]
    fallback_n = sum(1 for r in valid_logs
                     if 'fallback' in str(r.get('t_white_method', '')))
    feedback_n = sum(1 for r in valid_logs
                     if r.get('feedback_iterations', 0) > 0)

    print(f"\n{'='*72}")
    print(f"  TAMAMLANDI")
    print(f"  Başarılı                : {n_ok}")
    print(f"  Hata                    : {n_err}")
    print(f"  Fallback t_white kullanan: {fallback_n} dilim "
          f"(%{fallback_n/max(n_ok,1)*100:.1f})")
    print(f"  Feedback gereken         : {feedback_n} dilim "
          f"(%{feedback_n/max(n_ok,1)*100:.1f})")
    print(f"\n  Çıktılar:")
    print(f"    Traşlanmış PNG : {output_path}")
    print(f"    Halo mask PNG  : {halo_path}")
    print(f"    Excel istatistik: {excel_path}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()