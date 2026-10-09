# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              TRAŞLAMA ALGORİTMASI — SÜRÜM 4 (traslamaV4.py)               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  BU SCRIPT TEK SEFERDE ÜÇ ÇIKTI ÜRETIR:                                    ║
║                                                                              ║
║  1. TRAŞLANMIŞ GÖRÜNTÜLER  → OUTPUT_DIR/                                    ║
║     Kafatası pikselleri sıfırlanmış (0 = siyah), beyin korunmuş PNG'ler.   ║
║                                                                              ║
║  2. HALO MASKELERI          → OUTPUT_DIR/halo_mask_png/                     ║
║     Hangi piksellerin traşlandığını gösteren binary PNG'ler.                ║
║     Beyaz (255) = traşlanan alan, Siyah (0) = dokunulmayan alan.           ║
║     Sonraki istatistik veya Dice/IoU hesapları için kullanılır.             ║
║                                                                              ║
║  3. İSTATİSTİK EXCEL        → OUTPUT_DIR/v4_istatistik.xlsx                ║
║     Her dilim için: fg_pct, slice_class, t_white (ve yöntemi), max_depth,  ║
║     feedback bilgisi, halo piksel sayısı/oranı, beyin/kafatası yoğunluk    ║
║     istatistikleri (mean, std, p5...p95), gap analizi, hasta özeti.        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ÖNCEKI SÜRÜMLERDE SAPTANAN HATALAR VE V4'TE YAPILAN DÜZELTMELer:          ║
║                                                                              ║
║  BUG 1 — V3 MAX_DEPTH=20px → DİNAMİK TABLO                                  ║
║    Gerçek kafatası P90 = 107px (normal dilim), V3 hepsini 20'de kesiyordu. ║
║    Dilimlerin %37.8'i kafatası bitmeden duruyordu (ort. 59px eksik).        ║
║    → fg_pct'ye göre 25–147px arasında dinamik MAX_DEPTH.                    ║
║                                                                              ║
║  BUG 2 — V3 t_white [150-230] sabit pencere → FULL HISTOGRAM TARAMA        ║
║    find_peaks ile tüm histogram taranır. 3 güvenilirlik kriteri var;        ║
║    biri bile başarısız olursa V2'nin güvenli fallback'ine geçer.            ║
║                                                                              ║
║  BUG 3 — V3 thin dilim kilit (t_white > img_max) → GÜVENLIK KIRPMASI       ║
║    t_white = min(t_white, img_max - 5) her zaman uygulanır.                ║
║                                                                              ║
║  BUG 4 — V3 feedback t_white artırıyordu → V4 MAX_DEPTH artırır            ║
║    "Az traşladım" durumunda eşiğe dokunulmaz, derinlik 15px artırılır.     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

KULLANIM:
    python traslamaV4.py

GEREKSİNİMLER:
    pip install numpy pillow scipy pandas openpyxl
"""
#!!!!!!!!!!!! Sadece PEAK_MIN_SEPARATION_HU=40→12 yapıldı !!!!!!!!!!!!!!!!!!!!

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import binary_erosion
import warnings

warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        AYARLAR — BURADAN DEĞİŞTİR                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Orijinal (traşlanmamış) CT görüntüleri — beyin maskesi uygulanmış
INPUT_DIR = r"C:\Users\Yusuf Soylu\Desktop\Beyin Traslama\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\yedek\SiyahVeKaymaTemizBrainMaskUyg"
# Traşlanmış görüntülerin kaydedileceği ana klasör
# → Halo mask alt klasörü ve Excel de buraya kaydedilir
OUTPUT_DIR = r"C:\Users\Yusuf Soylu\Desktop\Beyin Traslama\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V4.1 deneme"

# GT (uzman) mask klasörü — istatistik exceli için beyin bölgesi yoğunlukları
# Eğer GT maskelerin yoksa veya istemiyorsan None yap:  GT_MASK_DIR = None
GT_MASK_DIR = None

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     TRAŞLAMA PARAMETRELERİ VE AÇIKLAMALARI                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Dilim boyutu sınırları ─────────────────────────────────────────────────────
VERY_THIN_MAX_FG_PCT = 2.0  # % — çok ince dilim
THIN_MAX_FG_PCT = 8.0  # % — ince dilim

# ── Dinamik MAX_DEPTH tablosu ──────────────────────────────────────────────────
"""
fg_pct grubuna giren tüm slice'ların skull_thickness_mean değerleri sıralanır, 90. yüzdelik dilim alınır, 
1.2 ile çarpılır → o grubun max_depth değeri elde edilir. Tablo bu işlemin 6 grup için tekrarından oluşur.
"""
# Kaynak: kafatasi_kalinlik_istatistik.xlsx → skull_thickness_mean P90 + %20 marj
# V3'te sabit 20px → dilimlerin %37.8'i yetersiz kalıyordu.
FG_TO_MAX_DEPTH_TABLE = [
    ( 2.0,  40),   # 0–2%    → very_thin  (49 slice)      very_thin: P90=20.9px → 25px
    ( 5.0,  83),   # 2–5%    → thin_low   (87 slice)      thin low:  P90=39.5px → 47px
    ( 8.0, 104),   # 5–8%    → thin_high  (105 slice)     thin high: P90=57.3px → 69px
    (15.0, 130),   # 8–15%   → normal_low (221 slice)     normal low:P90=61.7px → 74px
    (25.0, 160),   # 15–25%  → normal_mid (377 slice)     normal mid:P90=92.9px → 111px
    (35.0, 193),   # 25–35%  → normal_big (396 slice)     normal big:P90=122.2px→ 147px
    (100.0,200),   # 35–100% → çok büyük  (18 slice)      çok büyük (nadir): güvenlik tavanı
]

# ── t_white peak tespiti güvenilirlik kriterleri ────────────────────────────────
# Kaynak: histogram analizinde 432 dilimde t_white<100 yanlış çıktı.
# Bu üç kriter başarısız → fallback'e geç.
"""
PEAK_MIN_SEPARATION_HU:
peak_separation_hu (analiz scriptinde):
  Ort: ~45 HU
  P25: ~35 HU
  P50: ~45 HU
Bu değerlere bakılarak "40 HU mantıklı eşik" kararı verildi.

PEAK_SKULL_MIN_POS: Normalize edilmiş görüntüde kafatası kemiği en az 140 HU değerinde görünmeli. 
Bunun altında pik → bu kafatası değil, beyin veya geçiş bölgesi.

PEAK_VALLEY_MAX_RATIO: 
0.60 → "vadi en fazla beyin tepesinin %60'ı kadar yüksek olabilir." Bu eşiği geçen histogram düz → gerçek vadi yok → fallback.
valley_ratio = vadi yüksekliği / beyin tepesi yüksekliği
hu = 40 dediğimiz için zaten bu hiç tetiklenmedi
"""
PEAK_MIN_SEPARATION_HU = 12  # iki pik arası min mesafe (HU)
PEAK_SKULL_MIN_POS = 100  # kafatası piki min konumu. Bu değer literatüre göre seçilmiş
PEAK_VALLEY_MAX_RATIO = 0.60  # vadi derinlik oranı üst sınırı (0=derin, 1=düz)
PEAK_SMOOTHING_WINDOW = 11  # Savitzky-Golay pencere boyutu

# ── Fallback t_white ───────────────────────────────────────────────────────────
# V2 formülü: max(220, 0.92 × img_max) — en güvenilir sabit yöntem.
FALLBACK_T_WHITE_FACTOR = 0.92
FALLBACK_T_WHITE_MIN = 220

# ── Işın yürüyüşü ──────────────────────────────────────────────────────────────
# V1/V2/V3'te 4 yön vardı → köşe halolar kaçıyordu.
# 8 yön = 360°/8 = her 45°'de bir ışın. Daha fazla yön (16, 32) daha iyi ama daha yavaş. 8 denge noktası.
N_DIRECTIONS = 8  # 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

# ── Güvenlik sınırları ─────────────────────────────────────────────────────────
MIN_MAX_DEPTH = 8  # asla 8px altına inme
T_WHITE_MARGIN = 5  # t_white = min(t_white, img_max - T_WHITE_MARGIN)

# ── Feedback loop ──────────────────────────────────────────────────────────────
# Yeterli traşlama yoksa MAX_DEPTH artır (t_white'a dokunma — V3'ten fark)
FEEDBACK_TARGET_SHAVE_PCT = 0.5  # fg'nin en az bu %'si traşlanmalı
FEEDBACK_DEPTH_INCREMENT = 15  # her turda eklenen derinlik (px)
FEEDBACK_MAX_ITERATIONS = 4  # maksimum tekrar sayısı
FEEDBACK_MAX_DEPTH_CAP = 180  # ulaşılabilecek mutlak tavan (px)


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
    """fg_pct'ye göre very_thin / thin / normal sınıflandırması."""
    if fg_pct <= VERY_THIN_MAX_FG_PCT:
        return "very_thin"
    elif fg_pct <= THIN_MAX_FG_PCT:
        return "thin"
    return "normal"


def get_dynamic_max_depth(fg_pct: float) -> int:
    """
    fg_pct değerine göre bu dilim için MAX_DEPTH döndürür.
    FG_TO_MAX_DEPTH_TABLE tablosundan türetilmiştir.
    V3'te sabit 20px idi — artık 25-147px arası dinamik değer.
    """
    for fg_upper, depth in FG_TO_MAX_DEPTH_TABLE:
        if fg_pct <= fg_upper:
            return max(depth, MIN_MAX_DEPTH)  # MIN_MAX_DEPTH = 8  asla 8px altına inme
    return FEEDBACK_MAX_DEPTH_CAP  # FEEDBACK_MAX_DEPTH_CAP = 180   ulaşılabilecek mutlak tavan (px)


def _fallback_t_white(img_max: int) -> int:
    """
    Peak tespiti güvenilmez olduğunda kullanılan güvenli eşik.
    V2'nin formülü: max(220, 0.92 × img_max).
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

    3 güvenilirlik kontrolü geçilmezse fallback'e geçer.

    Döndürür: (t_white: int, method: str, debug: dict)
    """
    debug = {}
    fg_pixels = img[fg_mask].astype(np.int32)

    # Arka plan 50 pikselden az olursa. Hiç bir zaman tetiklenmiyor.
    if len(fg_pixels) < 50:
        return _fallback_t_white(img_max), "fallback_too_few_pixels", debug

    hist, _ = np.histogram(fg_pixels, bins=256, range=(0, 255))

    # Savitzky-Golay: gürültüyü temizler, pik konumlarını korur
    # Bu if her zaman true döner. Savunmacı programlama.
    if len(hist) >= PEAK_SMOOTHING_WINDOW:  # PEAK_SMOOTHING_WINDOW   = 11    # Savitzky-Golay pencere boyutu
        hist_s = savgol_filter(hist.astype(float),
                               window_length=PEAK_SMOOTHING_WINDOW,
                               polyorder=2)
        hist_s = np.clip(hist_s, 0, None)

    # Savgol çalışamazsa ham histogramı float'a çevir, düzleştirme yok. Pratikte hiç tetiklenmiyor.
    else:
        hist_s = hist.astype(float)

    """
    hist_s:
      bin 105 → 19500  ← yerel maksimum, ama...
      bin 120 → 20000  ← 15 bin uzakta, bu daha büyük
                          find_peaks distance=15 ile ikisi de pik
      bin 240 →  8000  ← kafatası tepesi
    peaks = [105, 120, 240]   ← 3 pik bulundu
    """

    # Tüm yerel maksimumları bul
    # distance=15: iki pik arası min 15 birim
    # prominence=10: çevresine göre en az 10 birim çıkıntılı
    peaks, _ = find_peaks(hist_s, distance=15, prominence=10)
    debug["n_peaks"] = len(peaks)

    # Tepe bulamazsa
    if len(peaks) < 2:
        return _fallback_t_white(img_max), "fallback_less_than_2_peaks", debug

    """
    peaks      = [105, 120, 240]
    hist_s[peaks] = [19500, 20000, 8000]

    np.argsort(hist_s[peaks]) : [19500, 20000, 8000] → argsort → [2, 0, 1]

    [2, 0, 1][-2:] = [0, 1]

    peaks[[0, 1]] = peaks[0] = 105  (19500 yükseklik),  peaks[1] = 120  (20000 yükseklik)
    """
    # En büyük 2 piki konuma göre sırala
    top2 = np.sort(peaks[np.argsort(hist_s[peaks])[-2:]])
    brain_peak = int(top2[0])  # düşük yoğunluk = beyin
    skull_peak = int(top2[1])  # yüksek yoğunluk = kafatası
    separation = skull_peak - brain_peak  # 120 - 105 = 15

    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Buradaki kritik sorun: En yüksek iki pik beyin ve kafatası olmak zorunda değil. Beynin gri madde/beyaz madde alt bantları 
    iki ayrı pik oluşturabilir — ikisi de beyin, kafatası atlanıyor. Separation=15 bunun göstergesi.
    """

    debug.update({"brain_peak": brain_peak, "skull_peak": skull_peak,
                  "separation": separation})

    # Kontrol 1: Pik ayrımı
    # separation < 40 → iki pik birbirine çok yakın → aynı dokunun alt bantları, gerçek beyin/kafatası ayrımı değil → fallback.
    # !!!!!!!Dataset'te: 1225 slice'ta separation=15 → 15 < 40 → fallback. Bu tek kontrol %97.8 fallback'in sebebi. !!!!!!!!!!
    if separation < PEAK_MIN_SEPARATION_HU:  # PEAK_MIN_SEPARATION_HU = 40
        debug["fail"] = f"sep={separation}"
        return _fallback_t_white(img_max), "fallback_peaks_too_close", debug

    # Kontrol 2: Kafatası piki konumu
    if skull_peak < PEAK_SKULL_MIN_POS:  # PEAK_SKULL_MIN_POS = 140
        debug["fail"] = f"skull={skull_peak}"
        return _fallback_t_white(img_max), "fallback_skull_peak_too_low", debug

    # İki pik arasındaki minimum = vadi noktası = t_white
    segment = hist_s[brain_peak: skull_peak + 1]
    valley_pos = brain_peak + int(np.argmin(segment))
    brain_h = hist_s[brain_peak]
    valley_h = hist_s[valley_pos]
    valley_ratio = (valley_h / brain_h) if brain_h > 0 else 1.0

    debug.update({"valley_pos": valley_pos,
                  "valley_ratio": round(valley_ratio, 3)})

    # Kontrol 3: Vadi derinliği
    # Vadi beyin tepesinin %60'ından yüksekse → düz histogram, gerçek vadi yok → fallback.
    # Dataset'te: Hiç tetiklenmedi. Kontrol 1 zaten her şeyi yakalıyor — separation=15 < 40 → fallback. Kontrol 3'e ulaşan neredeyse hiç slice yok.
    if valley_ratio > PEAK_VALLEY_MAX_RATIO:
        debug["fail"] = f"ratio={valley_ratio:.2f}"
        return _fallback_t_white(img_max), "fallback_shallow_valley", debug

    # Güvenlik kırpması — V3 thin bug önlemi
    t_white = int(valley_pos)
    # t_white hiçbir zaman img_max'ı geçemez
    t_white = min(t_white, img_max - T_WHITE_MARGIN)
    # Alt sınır. Beyin dokusu en az 120 HU değerinde görünür. 120'nin altında t_white → beyin dokusu da silinir. Güvenlik garantisi.
    t_white = max(t_white, 120)  # çok düşük eşik beyin yer

    return t_white, "two_peak_valley", debug


def find_fg_boundary(fg_mask: np.ndarray) -> list:
    """
    Ön plan maskesinin sınır piksellerini bulur (erozyon yöntemi).
    Işınlar buradan başlatılır.

    fg_mask:         eroded:          ~eroded:         fg & ~eroded:

    0 0 0 0 0        0 0 0 0 0        1 1 1 1 1        0 0 0 0 0
    0 1 1 1 0        0 0 0 0 0        1 1 1 1 1        0 1 1 1 0
    0 1 1 1 0   -    0 0 1 0 0   →    1 1 0 1 1   →    0 1 0 1 0
    0 1 1 1 0        0 0 0 0 0        1 1 1 1 1        0 1 1 1 0
    0 0 0 0 0        0 0 0 0 0        1 1 1 1 1        0 0 0 0 0
    """
    # Mantık: fg_mask=True ama eroded=False → bu piksel önplanda ama aşındırma onu sildi → demek ki sınır pikseliydi.

    eroded = binary_erosion(fg_mask)
    boundary = fg_mask & ~eroded  # ~ işareti tersleme yapar
    ys, xs = np.where(boundary)  # boundary'de True olan tüm piksellerin koordinatlarını döndürür
    return list(zip(ys.tolist(), xs.tolist()))


def ray_march(img: np.ndarray, start_y: int, start_x: int,
              dy: float, dx: float, t_white: int,
              max_depth: int, fg_mask: np.ndarray) -> np.ndarray:
    """
    img → orijinal görüntü
    start_y, start_x → ışının başladığı sınır pikseli koordinatı
    dy, dx → yön vektörü (normalleştirilmiş, örn. 45° için dy=0.707, dx=0.707)
    t_white → kafatası eşiği
    max_depth → maksimum adım sayısı
    fg_mask → önplan maskesi

    Tek yönde ışın yürütür ve traşlama maskesini döndürür.

    Adım adım:
      - Görüntü/ön plan sınırı aşıldı  → dur
      - max_depth aşıldı               → dur
      - piksel >= t_white              → kafatası, traşla ve devam
      - piksel < t_white               → beyin dokusu, dur

    Her yön kendi maskesini döndürür (V1'deki çakışma hatası önlenir).
    """
    h, w = img.shape
    mask = np.zeros((h, w), dtype=bool)
    depth = 0
    cy, cx = float(start_y), float(start_x)

    while True:
        # Çapraz yönlerde (45°) dy=0.707, dx=0.707 — her adımda 0.707 ekleniyor.

        cy += dy;
        cx += dx;
        depth += 1
        iy, ix = int(round(cy)), int(round(cx))

        if iy < 0 or iy >= h or ix < 0 or ix >= w:  break
        if not fg_mask[iy, ix]:                       break
        if depth > max_depth:                         break

        if img[iy, ix] >= t_white:
            mask[iy, ix] = True  # kafatası: traşla, devam et
        else:
            break  # beyin: dur

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

    """
    i=0: sin(0°)=0.000,  cos(0°)=1.000   → sağa (→)
    i=1: sin(45°)=0.707, cos(45°)=0.707  → sağ-aşağı (↘)
    i=2: sin(90°)=1.000, cos(90°)=0.000  → aşağı (↓)
    i=3: sin(135°)=0.707,cos(135°)=-0.707→ sol-aşağı (↙)
    i=4: sin(180°)=0.000,cos(180°)=-1.000→ sola (←)
    i=5: sin(225°)=-0.707,cos(225°)=-0.707→ sol-yukarı (↖)
    i=6: sin(270°)=-1.000,cos(270°)=0.000→ yukarı (↑)
    i=7: sin(315°)=-0.707,cos(315°)=0.707→ sağ-yukarı (↗)
    """
    # Yön vektörleri
    directions = [
        (np.sin(2 * np.pi * i / N_DIRECTIONS),  # N_DIRECTIONS = 8
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
                continue  # dışa giden yön
            final |= ray_march(img, sy, sx, dy, dx,  # final |= ...: Bitwise OR ile birleştir.
                               t_white, max_depth, fg_mask)
    return final

    """
    V1/V2/V3 — 4 yön, satır/sütun tabanlı:

    Her satır için:     Her sütun için:
    → → → ████          ↓ ↓ ████ ↑ ↑
    → → ████             ↓ ████ ↑
    → → → ████          ↓ ↓ ████ ↑ ↑

    Sadece yatay/dikey, köşe kaçıyor

    V4 — 8 yön, sınır tabanlı:

    ★ ★ ★ ★ ★
    ★ ████ ★       Her ★'dan 8 yön
    ★ ████ ★       Köşeler de yakalanıyor
    ★ ★ ★ ★ ★
    """


def apply_shave_with_feedback(img: np.ndarray, fg_mask: np.ndarray,
                              t_white: int, initial_max_depth: int,
                              fg_px: int) -> tuple:
    """
    Traşlama + feedback loop.

    Hedef traşlama oranı karşılanmazsa MAX_DEPTH artırılır.
    t_white'a kesinlikle dokunulmaz (V3'ten temel fark).

    Döndürür: (shave_mask, final_max_depth, n_iterations)
    """
    max_depth = initial_max_depth
    shave_mask = None

    for iteration in range(FEEDBACK_MAX_ITERATIONS + 1):  # FEEDBACK_MAX_ITERATIONS = 4     # maksimum tekrar sayısı
        # Tüm sınır piksellerinden 8 yönde ışın, OR ile birleştirme. Boolean maske döner.
        shave_mask = shave_slice(img, fg_mask, t_white, max_depth)
        # shave_mask.sum(): True olan piksel sayısı = silinen piksel sayısı.
        shave_pct = shave_mask.sum() / max(fg_px, 1) * 100.0

        if shave_pct >= FEEDBACK_TARGET_SHAVE_PCT:  # FEEDBACK_TARGET_SHAVE_PCT = 0.5   # fg'nin en az 5%'i traşlanmalı
            break  # yeterli

        # V3: "çok fazla sildim" → t_white artır (eşiği yukarı çek, daha az sil)
        # V4: "az sildim"        → max_depth artır (daha derin git, daha fazla sil)
        if iteration < FEEDBACK_MAX_ITERATIONS:  # FEEDBACK_MAX_ITERATIONS   = 4     # maksimum tekrar sayısı
            max_depth = min(max_depth + FEEDBACK_DEPTH_INCREMENT,
                            # FEEDBACK_DEPTH_INCREMENT = 15    # her turda eklenen derinlik (px). Yeterli traşlama yoksa 15 px daha derin git.
                            FEEDBACK_MAX_DEPTH_CAP)  # FEEDBACK_MAX_DEPTH_CAP    = 180   # ulaşılabilecek mutlak tavan (px)
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
    prefix örneği: 'halo' → 'halo_mean', 'halo_p95', vb.
    """
    if len(pixels) == 0:
        keys = ['count', 'mean', 'std', 'min',
                'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'max']
        return {f'{prefix}_{k}': np.nan for k in keys}

    return {
        f'{prefix}_count': len(pixels),
        f'{prefix}_mean': round(float(np.mean(pixels)), 2),
        f'{prefix}_std': round(float(np.std(pixels)), 2),
        f'{prefix}_min': int(np.min(pixels)),
        f'{prefix}_p5': float(np.percentile(pixels, 5)),
        f'{prefix}_p10': float(np.percentile(pixels, 10)),
        f'{prefix}_p25': float(np.percentile(pixels, 25)),
        f'{prefix}_p50': float(np.percentile(pixels, 50)),
        f'{prefix}_p75': float(np.percentile(pixels, 75)),
        f'{prefix}_p90': float(np.percentile(pixels, 90)),
        f'{prefix}_p95': float(np.percentile(pixels, 95)),
        f'{prefix}_max': int(np.max(pixels)),
    }


def build_excel(all_rows: list, output_excel: Path) -> None:
    """
    Dilim bazlı istatistikleri alır, 3 sheet'lik Excel dosyası üretir.

    Sheet 1 — Tüm Dilimler:
        Her satır = bir dilim. Traşlama parametreleri + yoğunluk istatistikleri.

    Sheet 2 — Hasta Özeti:
        patient_id bazında dilim sayısı, ortalama metrikler.

    Sheet 3 — Dilim Sınıfı Özeti:
        very_thin / thin / normal sınıflarında özet istatistikler.
    """
    df = pd.DataFrame(all_rows).sort_values('filename').reset_index(drop=True)

    # patient_id: dosya adındaki ilk sayısal kısım (örn. "02_slice0067" → "02")
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
        hasta_df = df.groupby('patient_id', as_index=False).agg(agg_valid)
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
            sinif_df = df.groupby('slice_class', as_index=False).agg(sinif_valid)
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
      1. Görüntüyü oku (PIL — Türkçe/boşluklu yollar için güvenli)
      2. t_bg → fg_mask → fg_pct → slice_class → max_depth
      3. t_white tespit et (histogram veya fallback)
      4. t_white güvenlik kırpması (thin bug önlemi)
      5. Traşlama + feedback loop
      6. Halo mask üret ve kaydet
      7. İstatistik hesapla (halo yoğunlukları + GT beyin yoğunlukları + gap)
    """
    log = {'filename': img_path.name}

    try:
        # ── 1. Görüntüyü oku ────────────────────────────────────────────────
        # PIL: cv2'nin Türkçe karakter sorununu önler
        pil_img = Image.open(img_path).convert("L")
        img = np.array(pil_img, dtype=np.uint8)
        h, w = img.shape
        total_px = h * w
        img_max = int(img.max())
        img_min = int(img.min())

        log.update({'img_max': img_max, 'img_min': img_min,
                    'total_px': total_px})

        # ── 2. Ön plan maskesi ve dilim sınıfı ──────────────────────────────
        t_bg = get_t_bg(img)
        fg_mask = img > t_bg
        fg_px = int(fg_mask.sum())
        fg_pct = fg_px / total_px * 100.0

        slice_class = classify_slice(fg_pct)
        max_depth = get_dynamic_max_depth(fg_pct)

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
        # V3 thin bug: img_max < t_white → traşlama=0 durumunu önler
        t_white = min(t_white, img_max - T_WHITE_MARGIN)  # T_WHITE_MARGIN = 5

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

        shaved_px = int(shave_mask.sum())
        shaved_pct = shaved_px / max(fg_px, 1) * 100.0

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
        # Traşlanan piksel = orijinalde var (>0) VE V4'te sıfır olmuş
        # "Sadece v4==0" demiyoruz çünkü orijinal arka plan da zaten 0'dı.
        halo_mask_bool = (img > 0) & (output_img == 0)
        halo_uint8 = halo_mask_bool.astype(np.uint8) * 255

        halo_px = int(halo_mask_bool.sum())
        halo_pct = halo_px / total_px * 100.0

        log.update({'halo_px': halo_px,
                    'halo_pct': round(halo_pct, 4)})

        _save_halo(halo_uint8, halo_dir / img_path.name)

        # ── 8. Halo yoğunluk istatistikleri ─────────────────────────────────
        # Traşlanan bölgedeki orijinal piksel yoğunlukları
        # → "kafatası dokusu mu yoksa beyin mi traşlandı?" sorusunun cevabı
        # Beklenti: halo_mean > 180 (kafatası yoğun, beyin ~80-150)
        halo_pixels = img[halo_mask_bool].astype(np.uint8)
        log.update(px_stats(halo_pixels, 'halo'))

        # ── 9. GT beyin istatistikleri (varsa) ──────────────────────────────
        # GT maskesi ile örtüşen CT piksellerinin yoğunluğu
        # gap = skull_p25 - brain_p95 → negatif ise beyin ve kafatası örtüşüyor
        if gt_dir_path is not None:
            gt_file = Path(gt_dir_path) / img_path.name
            if gt_file.exists():
                gt_bin = np.array(
                    Image.open(gt_file).convert("L"), dtype=np.uint8) > 127
                brain_px = img[gt_bin].astype(np.uint8)
                log.update(px_stats(brain_px, 'brain'))
                log['brain_mask_px'] = int(gt_bin.sum())
                log['brain_mask_pct'] = round(gt_bin.sum() / total_px * 100, 4)

                # Gap analizi: kafatası en düşük yoğunluk - beyin en yüksek yoğunluk
                # Pozitif gap → net ayrım var (iyi)
                # Negatif gap → örtüşme var (eşik belirsiz)
                skull_p25 = log.get('halo_p25', np.nan)
                brain_p95 = log.get('brain_p95', np.nan)
                skull_p5 = log.get('halo_p5', np.nan)

                if not (np.isnan(skull_p25) or np.isnan(brain_p95)):
                    gap = skull_p25 - brain_p95
                    log['gap'] = round(gap, 2)
                    log['gap_negative'] = bool(gap < 0)

                    # Alternatif t_white tahminleri (V5 için referans)
                    log['t_white_midpoint'] = round((skull_p25 + brain_p95) / 2, 1)
                if not np.isnan(skull_p5):
                    log['t_white_skull_p5m10'] = round(skull_p5 - 10, 1)
            else:
                log.update(px_stats(np.array([], dtype=np.uint8), 'brain'))
                log['brain_mask_px'] = np.nan
                log['brain_mask_pct'] = np.nan
                log['gap'] = np.nan
                log['gap_negative'] = np.nan

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
    """Binary halo mask PNG'yi kaydet."""
    Image.fromarray(halo_arr).save(str(out_path))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              ANA DÖNGÜ                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    INPUT_DIR içindeki tüm PNG'leri işler.
    3 çıktı üretir: traşlanmış görüntüler + halo maskeler + Excel istatistik.
    """
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    halo_path = output_path / "halo_mask_png"
    excel_path = output_path / "v4_istatistik.xlsx"

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

    print(f"\n{'=' * 72}")
    print(f"  TRAŞLAMA V4  |  {len(png_files)} dosya")
    print(f"  Giriş          : {INPUT_DIR}")
    print(f"  Traşlanmış çıktı: {OUTPUT_DIR}")
    print(f"  Halo mask      : {halo_path}")
    print(f"  Excel istatistik: {excel_path.name}")
    print(f"  GT mask        : {gt_dir or '(kullanılmıyor)'}")
    print(f"{'=' * 72}\n")

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
            print(f"  !! HATA: {img_path.name} — {log['error']}")
        else:
            # Traşlanmış görüntüyü kaydet
            Image.fromarray(output_img).save(str(output_path / img_path.name))
            n_ok += 1

            # Dikkat çekici durumları ekrana yaz
            iters = log.get('feedback_iterations', 0)
            method = log.get('t_white_method', '')
            if i <= 3 or iters > 0 or 'fallback' in str(method):
                print(f"    {img_path.name}: "
                      f"{log.get('slice_class', '?'):9s} | "
                      f"t_white={log.get('t_white', '?'):3} "
                      f"({str(method)[:18]:18s}) | "
                      f"depth={log.get('max_depth_final', '?'):3}px | "
                      f"traş={log.get('shaved_pct_of_fg', 0):5.1f}% | "
                      f"halo={log.get('halo_px', 0):5}px"
                      + (f" [FB×{iters}]" if iters > 0 else ""))

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

    print(f"\n{'=' * 72}")
    print(f"  TAMAMLANDI")
    print(f"  Başarılı                : {n_ok}")
    print(f"  Hata                    : {n_err}")
    print(f"  Fallback t_white kullanan: {fallback_n} dilim "
          f"(%{fallback_n / max(n_ok, 1) * 100:.1f})")
    print(f"  Feedback gereken         : {feedback_n} dilim "
          f"(%{feedback_n / max(n_ok, 1) * 100:.1f})")
    print(f"\n  Çıktılar:")
    print(f"    Traşlanmış PNG : {output_path}")
    print(f"    Halo mask PNG  : {halo_path}")
    print(f"    Excel istatistik: {excel_path}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()