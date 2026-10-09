"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            BEYİN TRAŞLAMA — VERİ GÜDÜMLÜ UYARLAMALI v3.0                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  v2'den farklar — 5 temel yenilik:                                           ║
║                                                                              ║
║  YENİLİK 1 — Histogram Vadi Tespiti (t_white için):                         ║
║    v2: t_white = max(220, 0.92 × img_max) → sabit, veriye kör               ║
║    v3: Piksel histogramındaki beyin (~120) ve kafatası (~250) tepeleri       ║
║        arasındaki VADİ NOKTASI otomatik bulunur → her dilim için özel        ║
║        t_white. Veri setinden ölçülen median ideal: ~199.                    ║
║                                                                              ║
║  YENİLİK 2 — Dinamik Derinlik Sınırı (MAX_STRIP_DEPTH):                    ║
║    v2: 25 px sabit — veri setinde P99 gerçek derinlik = 16 px               ║
║    v3: depth = max(MIN_DEPTH, int(sqrt(non_bg_pixels) × DEPTH_RATIO))       ║
║        Küçük beyin → az derinlik, büyük beyin → biraz daha fazla            ║
║                                                                              ║
║  YENİLİK 3 — Adaptif İnce Dilim Eşiği:                                     ║
║    v2: %5 sabit eşik — veri seti analizi %8'in daha iyi olduğunu gösterdi  ║
║    v3: non_bg_pct < THIN_SLICE_THRESHOLD (%8) → 3 aşamalı dilim sınıfı     ║
║        Normal / İnce / Çok İnce → her sınıf için ayrı t_white stratejisi   ║
║                                                                              ║
║  YENİLİK 4 — Geri Besleme Döngüsü (Feedback Loop):                         ║
║    v2: Tek geçiş — yanlış t_white ile tıraşlanınca düzeltme yok             ║
║    v3: Tıraşlama sonrası kalite kontrolü:                                    ║
║        shave/brain > SAFETY_RATIO → t_white'ı FEEDBACK_STEP artır, tekrar  ║
║        En fazla MAX_FEEDBACK_ITER iterasyon → aşırı tıraşlama otomatik engel║
║                                                                              ║
║  YENİLİK 5 — Zenginleştirilmiş İstatistik Çıktısı:                         ║
║    v3: Histogram analizi, vadi noktası, kaç iterasyon gerekti,               ║
║        geri besleme tetiklenip tetiklenmediği, güven skoru → Excel'e         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Veri seti analizi sonuçları (1296 dilim, 35 hasta):
  ┌─────────────────────────────────────────────────────────────┐
  │ Kafatası min yoğunluğu  : ort=237, medyan=235              │
  │ Beyin p95 yoğunluğu     : ort=165, medyan=161              │
  │ Gap (kafatası-beyin_p95): ort=71,  medyan=74               │
  │ Gap < 0 (örtüşme)       : 24 dilim (%1.9) → ort. yenme %44│
  │ Gerçek tıraşlama P99    : 16 px  → v2'nin 25 sınırı fazla │
  │ İnce dilim <%8          : 255 dilim, ort. yenme %12.7      │
  └─────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# >>> KLASÖR YOLLARINI BURADAN DEĞİŞTİR
# ============================================================================
INPUT_DIR  = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\SiyahVeKaymaTemizBrainMaskUyg"
OUTPUT_DIR = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\PNG data\Tekrarsız_Traşlanmış_Beyin_v3_EREN"

# ============================================================================
# GLOBAL PARAMETRELER — Bunlar artık "başlangıç noktası", resim bunları ezer
# ============================================================================

# ── Histogram Vadi Tespiti ───────────────────────────────────────────────────
# Histogram düzleştirme için pencere genişliği (aşırı gürültüyü bastırır)
HIST_SMOOTH_WINDOW  = 5

# Vadi araması için yoğunluk aralığı — beyin ve kafatası tepeleri bu aralıkta
HIST_VALLEY_LOW     = 150   # Bu değerin altını arama (beyin tepesi ~120)
HIST_VALLEY_HIGH    = 230   # Bu değerin üstünü arama (kafatası tepesi ~250)

# Histogram vadisi bulunamazsa yedek: max(FALLBACK_FLOOR, FALLBACK_FACTOR × img_max)
FALLBACK_FLOOR      = 225
FALLBACK_FACTOR     = 0.92

# ── Dinamik Derinlik ─────────────────────────────────────────────────────────
# depth = max(MIN_DEPTH, int(sqrt(non_bg_pixels) × DEPTH_RATIO))
# Veri: P99 gerçek derinlik = 16 px. sqrt(ortalama beyin pikseli ≈ 45000) × 0.06 ≈ 13 px
MIN_DEPTH           = 8     # Hiçbir zaman bunun altına inme
MAX_DEPTH           = 20    # Hiçbir zaman bunun üstüne çıkma (v2=25, burada daha güvenli)
DEPTH_RATIO         = 0.06  # sqrt(non_bg_pixels) × bu oran = tahmini derinlik

# ── Adaptif İnce Dilim Sınıfları ─────────────────────────────────────────────
# Üç sınıf: Normal / İnce / Çok İnce
# Veri seti analizi: %8 eşiği %5'ten daha iyi ayrım yapıyor
THIN_THRESHOLD      = 0.08   # %8 → İnce dilim (v2: %5)
VERY_THIN_THRESHOLD = 0.02   # %2 → Çok ince dilim

# İnce/çok ince dilimlerde t_white yedeği (vadi tespit edilemezse)
THIN_T_WHITE        = 248    # İnce dilim yedek eşiği
VERY_THIN_T_WHITE   = 252    # Çok ince dilim yedek eşiği (neredeyse hiç silme)

# ── Geri Besleme Döngüsü ─────────────────────────────────────────────────────
# Tıraşlama/Beyin oranı bu değeri geçerse → t_white artır, tekrar dene
SAFETY_RATIO        = 0.20   # %20 → "Beynin %20'sinden fazlasını sildim, dikkat!"
FEEDBACK_STEP       = 3      # Her iterasyonda t_white bu kadar artar
MAX_FEEDBACK_ITER   = 4      # En fazla 4 deneme (5. denemede sonucu kabul et)

# ── Arka Plan Eşiği ──────────────────────────────────────────────────────────
# v2 ile aynı — veri setinde iyi çalışıyor
BORDER_WIDTH        = 5      # Kenardan kaç piksel arka plan örneklenir
T_BG_PERCENTILE     = 99.9   # Kenar piksellerinin bu yüzdeliği = t_bg

# ============================================================================

import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI — Histogram düzleştirme
# ─────────────────────────────────────────────────────────────────────────────
"""
Değer 0   → 198.000 piksel (arka plan)
Değer 80  → 1.200 piksel
Değer 120 → 3.400 piksel  ← beyin tepesi
Değer 121 → 80 piksel     ← ANİ DİP (gürültü)
Değer 122 → 3.200 piksel
Değer 200 → 50 piksel     ← vadi bölgesi
Değer 240 → 2.800 piksel  ← kafatası tepesi

121 değerinde ani dip var. Ham histogramda bu gerçek vadi gibi görünür. Ama değil — sadece o değerde az piksel var, gürültü.
window=5 demek: her bin için kendisi + 2 solundaki + 2 sağındaki 5 binin ortalamasını al.

Ham değerler:
  bin[119] = 3.100
  bin[120] = 3.400   ← beyin tepesi
  bin[121] = 80      ← ani dip (gürültü)
  bin[122] = 3.200
  bin[123] = 3.000

Düzleştirilmiş bin[121] = (3100 + 3400 + 80 + 3200 + 3000) / 5 = 2.556

full → çıktı girişten uzun, kenarlar hesaba katılıyor
same → çıktı girişle aynı uzunlukta (256 bin → 256 bin)
valid → sadece tam örtüşen kısım

same seçildi çünkü histogram 256 bin, çıktı da 256 bin olmalı. Sonra hist_s[150:231] diye arama yapılıyor — 
indeksler bozulmaması için boyut aynı kalmalı.

Kenar Etkisi
mode='same' kullanınca histogramın başı ve sonu için komşu eksik. NumPy bunu sıfırla doldurur:
bin[0] için pencere: [0, 0, hist[0], hist[1], hist[2]] / 5
Ama bu problem değil — zaten 150–230 arasında arama yapılıyor, kenarlar o aralığa girmiyor.

Neden window=5?
Daha büyük window → daha fazla düzleştirme ama gerçek vadi de kaybolabilir. Daha küçük → gürültü bastırılmıyor. 5 bu dataset için orta yol. Teorik olarak şöyle düşün:

window=1 → düzleştirme yok, ham histogram
window=5 → hafif düzleştirme, küçük gürültüler yok
window=20 → aşırı düzleştirme, gerçek vadi de silinir
"""

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Basit kayan ortalama ile histogram düzleştirme."""
    if window < 2:
        return arr.astype(float)
    kernel = np.ones(window) / window       # [0.2, 0.2, 0.2, 0.2, 0.2]
    return np.convolve(arr.astype(float), kernel, mode='same')


# ─────────────────────────────────────────────────────────────────────────────
# YENİLİK 1 — Histogram Vadi Tespiti
# ─────────────────────────────────────────────────────────────────────────────

def find_valley_threshold(img_u8: np.ndarray, t_bg: int) -> tuple[int, bool]:
    """
    Arka plan hariç piksel histogramında, HIST_VALLEY_LOW–HIST_VALLEY_HIGH (150-230)
    aralığındaki MİNİMUM noktayı (vadiyi) bul → t_white olarak döndür.

    Mantık:
        Beyin dokusu histogramda ~80–180 arasında bir tepe oluşturur.
        Kafatası kemiği histogramda ~230–255 arasında başka bir tepe oluşturur.
        İki tepe arasındaki minimum = en iyi ayırma noktası (Otsu benzeri).

    Döndürür:
        (t_white, valley_found)
        valley_found=False → fallback değeri kullanıldı
    """
    """
    Arka plan (hava)    : 0–5     → 198.000 piksel
    Beyin dokusu        : 70–160  → 45.000 piksel
    Kafatası kemiği     : 220–255 → 8.000 piksel
    Ara bölge           : 160–220 → az piksel (geçiş)

    Piksel  |  Sayı
    --------|--------
      70    |  200
      80    |  800
      90    | 2.100
     100    | 4.500
     110    | 6.800   ← BEYİN TEPESİ
     120    | 7.200   ← BEYİN TEPESİ (en yüksek)
     130    | 6.100
     140    | 4.200
     -------------------------------------------------
     150    | 2.800
     160    | 1.400
     170    |   900
     180    |   600
     190    |   350
     200    |   180   ← VADİ BÖLGESI (az piksel)
     210    |   220
     220    |   800
     230    | 2.400
     -------------------------------------------------
     240    | 3.600   ← KAFATASI TEPESİ
     250    | 3.200
     255    | 1.100
     
     120 civarında beyin tepesi, 200 civarında dip, 240 civarında kafatası tepesi. Vadi 200 civarında.
    """
    # Arka plan piksellerini dışla. t_bg = 0
    # Neden? Arka plan histogramı baskın tepe oluşturur, gerçek vadi arama bölgesini bozar.
    fg = img_u8[img_u8 > t_bg]
    if len(fg) == 0:
        return FALLBACK_FLOOR, False   # FALLBACK_FLOOR = 225

    # 256 değerlik histogram (0–255)
    hist, _ = np.histogram(fg, bins=256, range=(0, 255))
    # Ham histogramdaki küçük dalgalanmalar yumuşatıldı. Örneğin 190=350, 200=180, 210=220 iken düzleştirme sonrası bunlar birbirine yaklaşır.
    # Gerçek yapı bozulmaz.
    hist_s   = _smooth(hist, HIST_SMOOTH_WINDOW)

    # Arama aralığı: HIST_VALLEY_LOW … HIST_VALLEY_HIGH   150-230 arası
    lo = max(0, HIST_VALLEY_LOW)        # 0-150
    hi = min(255, HIST_VALLEY_HIGH)     # 230-255
    if lo >= hi:
        return _fallback_t_white(img_u8), False

    """
    Beyin tepesi ~120'de → 150'nin altına bakma, beyin tepesini vadi zannetme
    Kafatası tepesi ~240'da → 230'un üstüne bakma, kafatası tepesini vadi zannetme
    Vadi bu ikisi arasında, yani 150–230 arası
    """
    region = hist_s[lo:hi + 1]

    # Arama aralığında hiç piksel yoksa → fallback
    if region.sum() == 0:
        return _fallback_t_white(img_u8), False

    # Minimum noktayı bul
    # region içindeki minimum 200 değerinde (180 sayım). argmin → bölgenin 50. elemanı (200-150=50). valley = 150 + 50 = 200.
    valley_offset = int(np.argmin(region))
    valley        = lo + valley_offset

    """
    150 → 300
    160 → 280
    170 → 260   ← "minimum" ama bu düz iniş, tepe yok
    180 → 250
    
    Burada minimum var ama iki tarafta net tepe yok — bu görüntüde kafatası kemiği yok, sadece beyin var.
    """

    # Doğrulama: Vadi, her iki tepe değerinin en az %5'inden küçük olmalı
    left_max  = float(hist_s[:valley].max()) if valley > 0 else 1.0
    right_max = float(hist_s[valley + 1:].max()) if valley < 255 else 1.0
    valley_val = float(hist_s[valley])

    """
    left_max  = 7.200  (beyin tepesi, 120. bin)
    right_max = 3.600  (kafatası tepesi, 240. bin)
    valley_val = 180   (200. bin)
    
    180 < 0.95 × 7.200 = 6.840  ✓
    180 < 0.95 × 3.600 = 3.420  ✓
    7.200 > 10                   ✓
    3.600 > 10                   ✓
    
    → is_real_valley = True ✓
    """

    is_real_valley = (
        valley_val < 0.95 * left_max and    # vadi sol tepeden %5 düşük
        valley_val < 0.95 * right_max and   # vadi sağ tepeden %5 düşük
        left_max > 10 and                   # sol tarafta anlamlı tepe var
        right_max > 10                      # sağ tarafta anlamlı tepe var
    )

    # Sıkıntı yoksa devam et. Sıkıntı varsa _fallback_t_white
    if is_real_valley:
        return valley, True
    else:
        return _fallback_t_white(img_u8), False

    """
    Ne Zaman Vadi Bulunamıyor?
    Senaryo: thin/very_thin dilim
    
    Piksel  | Sayı
    --------|------
      80    |  10
     100    |  15
     120    |  12   ← çok az piksel, net tepe yok
     200    |   3
     240    |   8   ← çok az piksel, net tepe yok
     
    İki tepe de çok küçük → right_max > 10 koşulu sağlanamıyor → vadi bulunamadı → fallback. Dataset'te thin/very_thin 241 slice'ta hiç vadi bulunamadı, zaten beklenen sonuç.
    Senaryo: kafatası kemiği çok az olan slice
    
    150 → 100
    160 →  80
    170 →  60
    180 →  40   ← minimum
    190 →  50
    200 →  60
    ...
    
    240 →   5   ← right_max=5, 10'dan küçük
    right_max > 10 sağlanamıyor → fallback.
    """


def _fallback_t_white(img_u8: np.ndarray) -> int:
    """Histogram vadisi bulunamazsa yedek t_white hesapla."""
    mx = int(img_u8.max())
    return int(max(FALLBACK_FLOOR, FALLBACK_FACTOR * mx))


# ─────────────────────────────────────────────────────────────────────────────
# YENİLİK 2+3 — Tam Eşik Hesaplama (t_bg, t_white, derinlik, dilim sınıfı)
# ─────────────────────────────────────────────────────────────────────────────

def compute_thresholds(img_u8: np.ndarray) -> dict:
    """
    Görüntüye özel tüm parametreleri hesapla.

    Döndürür:
        t_bg         : arka plan eşiği
        t_white      : tıraşlama eşiği (histogramdan)
        max_depth    : bu dilim için maksimum tıraşlama derinliği
        slice_class  : 'normal' | 'thin' | 'very_thin'
        non_bg_pct   : non-background piksel oranı (0–1)
        non_bg_pixels: non-background piksel sayısı
        valley_found : histogram vadisi bulundu mu
        img_max      : görüntü maksimum değeri
        img_p95      : görüntü 95. yüzdeliği
    """
    h, w = img_u8.shape

    # ── t_bg: kenar piksellerinden arka plan eşiği ───────────────────────────
    bw = BORDER_WIDTH
    border = np.concatenate([
        img_u8[:bw,  :].ravel(),
        img_u8[-bw:, :].ravel(),
        img_u8[:,  :bw].ravel(),
        img_u8[:, -bw:].ravel()
    ])
    t_bg = int(np.percentile(border, T_BG_PERCENTILE))

    # ── Non-background piksel analizi ────────────────────────────────────────
    # Arka paln olmayan nesnenin tüm resme oranı
    fg_mask      = img_u8 > t_bg
    non_bg_px    = int(fg_mask.sum())
    non_bg_pct   = non_bg_px / img_u8.size

    # ── Dilim sınıfı (YENİLİK 3) ─────────────────────────────────────────────
    if non_bg_pct < VERY_THIN_THRESHOLD:
        slice_class = 'very_thin'
    elif non_bg_pct < THIN_THRESHOLD:
        slice_class = 'thin'
    else:
        slice_class = 'normal'

    # ── t_white: histogram vadi tespiti (YENİLİK 1) ──────────────────────────
    if slice_class == 'very_thin':
        # Çok ince dilim: neredeyse hiç silme
        t_white      = VERY_THIN_T_WHITE       # VERY_THIN_T_WHITE   = 252 Dataset'te ort. 252.2.
        valley_found = False                   # Vadi Yok
    elif slice_class == 'thin':
        """
        thin: Önce vadi ara. Ama vadi < THIN_T_WHITE (248) çıkarsa → 248 kullan. Dataset'te thin'de vadi hiç bulunamadı 
        (0/192) → tüm thin slice'lar t_white=248 kullandı.
        """
        # İnce dilim: önce vadi ara, bulamazsan sabit yedek
        t_white, valley_found = find_valley_threshold(img_u8, t_bg)
        # İnce dilimde vadi çok düşük çıkarsa yedek kullan
        if t_white < THIN_T_WHITE:
            t_white      = THIN_T_WHITE     # THIN_T_WHITE = 248
            valley_found = False            # Vadi Yok
    else:
        # Normal dilim: tam histogram vadi tespiti
        t_white, valley_found = find_valley_threshold(img_u8, t_bg)

    # ── Dinamik derinlik (YENİLİK 2) ─────────────────────────────────────────
    # sqrt(non_bg_pixels) orantılı — büyük beyin = biraz daha derinlik izni
    # MIN_DEPTH = 8
    # MAX_DEPTH = 20
    # DEPTH_RATIO = 0.06
    raw_depth = int(np.sqrt(max(non_bg_px, 1)) * DEPTH_RATIO)
    # clip ile MIN_DEPTH <= olan değerler MIN_DEPTH olur. MAX_DEPTH >= olan değerler de MAX_DEPTH olur
    max_depth = int(np.clip(raw_depth, MIN_DEPTH, MAX_DEPTH))

    # Çok ince dilimlerde derinliği daha da kısıt
    if slice_class == 'very_thin':
        max_depth = MIN_DEPTH
    elif slice_class == 'thin':
        max_depth = min(max_depth, MIN_DEPTH + 2)   #  min(hesaplanan, 10)

    return {
        't_bg':          t_bg,
        't_white':       t_white,
        'max_depth':     max_depth,
        'slice_class':   slice_class,
        'non_bg_pct':    non_bg_pct,
        'non_bg_pixels': non_bg_px,
        'valley_found':  valley_found,
        'img_max':       int(img_u8.max()),
        'img_p95':       float(np.percentile(img_u8, 95)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CORE — Tek geçiş ray-marching (verilen parametrelerle)
# ─────────────────────────────────────────────────────────────────────────────

def _ray_march_pass(
    img_u8:    np.ndarray,
    t_bg:      int,
    t_white:   int,
    max_depth: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    4 yönlü bağımsız ray-marching (v2 mimarisi korundu).

    Döndürür:
        halo_mask : tıraşlanan bölge maskesi (np.uint8)
        dir_pixels: yön bazlı piksel sayıları (dict)
    """
    h, w = img_u8.shape

    mask_L = np.zeros((h, w), dtype=np.uint8)
    mask_R = np.zeros((h, w), dtype=np.uint8)
    mask_T = np.zeros((h, w), dtype=np.uint8)
    mask_B = np.zeros((h, w), dtype=np.uint8)

    # ── Sol → Sağ ─────────────────────────────────────────────────────────────
    for y in range(h):
        x = 0
        while x < w and img_u8[y, x] <= t_bg:
            x += 1
        depth = 0
        while x < w and img_u8[y, x] >= t_white and depth < max_depth:
            mask_L[y, x] = 255
            x     += 1
            depth += 1

    # ── Sağ → Sol ─────────────────────────────────────────────────────────────
    for y in range(h):
        x = w - 1
        while x >= 0 and img_u8[y, x] <= t_bg:
            x -= 1
        depth = 0
        while x >= 0 and img_u8[y, x] >= t_white and depth < max_depth:
            mask_R[y, x] = 255
            x     -= 1
            depth += 1

    # ── Yukarı → Aşağı ────────────────────────────────────────────────────────
    for x in range(w):
        y = 0
        while y < h and img_u8[y, x] <= t_bg:
            y += 1
        depth = 0
        while y < h and img_u8[y, x] >= t_white and depth < max_depth:
            mask_T[y, x] = 255
            y     += 1
            depth += 1

    # ── Aşağı → Yukarı ────────────────────────────────────────────────────────
    for x in range(w):
        y = h - 1
        while y >= 0 and img_u8[y, x] <= t_bg:
            y -= 1
        depth = 0
        while y >= 0 and img_u8[y, x] >= t_white and depth < max_depth:
            mask_B[y, x] = 255
            y     -= 1
            depth += 1

    halo_mask = ((mask_L | mask_R | mask_T | mask_B) > 0).astype(np.uint8) * 255

    dir_pixels = {
        'left':   int((mask_L > 0).sum()),
        'right':  int((mask_R > 0).sum()),
        'top':    int((mask_T > 0).sum()),
        'bottom': int((mask_B > 0).sum()),
    }

    return halo_mask, dir_pixels


# ─────────────────────────────────────────────────────────────────────────────
# YENİLİK 4 — Geri Besleme Döngüsü ile Tıraşlama
# ─────────────────────────────────────────────────────────────────────────────

def strip_halo_v3(img_u8: np.ndarray) -> tuple:
    """
    Veri güdümlü uyarlamalı tıraşlama.

    Akış:
        1. compute_thresholds() → t_bg, t_white, max_depth, slice_class
        2. _ray_march_pass() → halo_mask
        3. Kalite kontrolü: shave/brain > SAFETY_RATIO?
              Evet → t_white += FEEDBACK_STEP, 2'ye dön (max MAX_FEEDBACK_ITER kez)
              Hayır → kabul et
        4. Zenginleştirilmiş istatistik döndür

    Döndürür:
        cleaned        : tıraşlanmış görüntü (np.uint8)
        halo_mask      : tıraşlanan bölge maskesi (np.uint8)
        params         : compute_thresholds() sözlüğü (son iterasyondaki değerler)
        stats          : { dir_pixels, feedback_triggered, n_iterations, confidence }
    """
    # Adım 1: Başlangıç parametreleri
    params = compute_thresholds(img_u8)

    t_bg      = params['t_bg']
    t_white   = params['t_white']
    max_depth = params['max_depth']
    non_bg_px = params['non_bg_pixels']

    feedback_triggered = False
    n_iter             = 1

    # Adım 2+3: Tıraşlama + Geri besleme döngüsü
    # MAX_FEEDBACK_ITER = 4
    """
    Mantık: Silinen piksel / non-background piksel oranı > %20 ise "çok fazla sildim" diye t_white'ı 3 artır, tekrar taşla. 
    Bu oran güvenli bölgeye inene kadar veya 4 iterasyona ulaşana kadar devam et.
    """
    for iteration in range(MAX_FEEDBACK_ITER + 1):
        halo_mask, dir_pixels = _ray_march_pass(img_u8, t_bg, t_white, max_depth)

        # shave_px: Traşlanan piksel sayısı
        shave_px = int((halo_mask > 0).sum())

        # Kalite kontrolü: tıraşlanan / non-background oranı
        # ratio = Traşlanan alan ile resimde arka plan olmayan alanının oranı
        ratio = shave_px / max(non_bg_px, 1)

        # # SAFETY_RATIO = 0.20  # %20 → "Beynin %20'sinden fazlasını sildim, dikkat!"
        if ratio <= SAFETY_RATIO or iteration == MAX_FEEDBACK_ITER:
            # Kabul et: oran güvenli VEYA iterasyon limiti doldu
            break

        # Aşırı tıraşlama → geri besleme
        feedback_triggered = True
        t_white           += FEEDBACK_STEP      # FEEDBACK_STEP = 3  # Her iterasyonda t_white bu kadar artar
        n_iter            += 1

        # t_white 255'i geçemez
        if t_white >= 255:
            t_white = 255
            break

    # Son kullanılan t_white'ı params'a yaz
    params['t_white']       = t_white
    params['t_white_final'] = t_white
    params['t_white_init']  = params['t_white']   # ilk hesaplanan (geri besleme öncesi)

    # Adım 4: Temizlenmiş görüntü
    cleaned = img_u8.copy()
    cleaned[halo_mask > 0] = 0

    # Güven skoru: histogram vadisi bulunduysa ve geri besleme tetiklenmediyse yüksek
    # high: Vadi bulundu + geri besleme yok → 862 slice (%68.8)
    # medium: Sadece biri → 381 slice (%30.4)
    # low: İkisi de kötü → 10 slice (%0.8) — bunlar geri besleme tetiklenen slice'ların tamamı
    valley_found = params['valley_found']
    if valley_found and not feedback_triggered:
        confidence = 'high'
    elif valley_found or not feedback_triggered:
        confidence = 'medium'
    else:
        confidence = 'low'

    stats = {
        'dir_pixels':          dir_pixels,
        'feedback_triggered':  feedback_triggered,
        'n_iterations':        n_iter,
        'confidence':          confidence,
            'shave_ratio':         round(ratio, 4),
    }

    return cleaned, halo_mask, params, stats


# ─────────────────────────────────────────────────────────────────────────────
# WORKER — tek dosya işle
# ─────────────────────────────────────────────────────────────────────────────

def process_file(file: Path, clean_dir: Path, mask_dir: Path) -> dict:
    try:
        img = np.array(Image.open(file).convert("L"), dtype=np.uint8)
        h, w = img.shape
        total_px = img.size

        cleaned, halo_mask, params, stats = strip_halo_v3(img)

        # Kaydet
        Image.fromarray(cleaned).save(clean_dir  / file.name)
        Image.fromarray(halo_mask).save(mask_dir / file.name)

        halo_count   = int((halo_mask > 0).sum())
        dir_px       = stats['dir_pixels']
        dir_dominant = max(dir_px, key=dir_px.get) if halo_count > 0 else 'none'
        dir_imbalance = round(
            max(dir_px.values()) / (sum(dir_px.values()) + 1e-9), 4
        ) if halo_count > 0 else 0.0

        return {
            # ── Dosya bilgisi ─────────────────────────────────────
            'filename':            file.name,
            'height':              h,
            'width':               w,
            'total_pixels':        total_px,

            # ── Ham görüntü istatistikleri ────────────────────────
            'img_min':             int(img.min()),
            'img_max':             params['img_max'],
            'img_mean':            round(float(img.mean()), 3),
            'img_std':             round(float(img.std()),  3),
            'img_p95':             params['img_p95'],

            # ── Eşik parametreleri (v3'e özgü) ───────────────────
            't_bg':                params['t_bg'],
            't_white_init':        params['t_white_init'],   # histogram vadisi
            't_white_final':       params['t_white_final'],  # geri besleme sonrası
            'valley_found':        params['valley_found'],   # YENİLİK 1
            'slice_class':         params['slice_class'],    # YENİLİK 3
            'non_bg_pct':          round(params['non_bg_pct'] * 100, 4),
            'non_bg_pixels':       params['non_bg_pixels'],
            'max_depth_used':      params['max_depth'],      # YENİLİK 2

            # ── Geri besleme (YENİLİK 4) ─────────────────────────
            'feedback_triggered':  stats['feedback_triggered'],
            'n_iterations':        stats['n_iterations'],
            'final_shave_ratio':   stats['shave_ratio'],
            'confidence':          stats['confidence'],      # YENİLİK 5

            # ── Tıraşlama sonuçları ───────────────────────────────
            'halo_pixels_removed': halo_count,
            'halo_pct':            round(halo_count / total_px * 100, 4),
            'dir_left_px':         dir_px['left'],
            'dir_right_px':        dir_px['right'],
            'dir_top_px':          dir_px['top'],
            'dir_bottom_px':       dir_px['bottom'],
            'dir_dominant':        dir_dominant,
            'dir_imbalance':       dir_imbalance,

            'error': '',
        }

    except Exception as e:
        blank = {k: 0 for k in [
            'height','width','total_pixels','img_min','img_max','img_mean','img_std',
            'img_p95','t_bg','t_white_init','t_white_final','non_bg_pct','non_bg_pixels',
            'max_depth_used','final_shave_ratio','halo_pixels_removed','halo_pct',
            'dir_left_px','dir_right_px','dir_top_px','dir_bottom_px','dir_imbalance',
            'n_iterations',
        ]}
        blank.update({
            'filename': file.name, 'error': str(e),
            'valley_found': False, 'slice_class': 'error',
            'feedback_triggered': False, 'confidence': 'error',
            'dir_dominant': 'error',
        })
        return blank


# ─────────────────────────────────────────────────────────────────────────────
# ANA PROGRAM
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print('=' * 68)
    print('  🧠  BEYİN TRAŞLAMA v3.0 — VERİ GÜDÜMLÜ UYARLAMALI')
    print('=' * 68)
    print()
    print('  [YENİLİK 1] Histogram Vadi Tespiti:')
    print(f'    Arama aralığı      : {HIST_VALLEY_LOW}–{HIST_VALLEY_HIGH}')
    print(f'    Düzleştirme pencere: {HIST_SMOOTH_WINDOW}')
    print(f'    Yedek (vadi yok)   : max({FALLBACK_FLOOR}, {FALLBACK_FACTOR}×max)')
    print()
    print('  [YENİLİK 2] Dinamik Derinlik:')
    print(f'    Formül             : max({MIN_DEPTH}, int(√non_bg × {DEPTH_RATIO}))')
    print(f'    Aralık             : [{MIN_DEPTH}, {MAX_DEPTH}] px  (v2: 25 sabit)')
    print()
    print('  [YENİLİK 3] Adaptif İnce Dilim:')
    print(f'    Çok ince           : non_bg < %{VERY_THIN_THRESHOLD*100:.0f} → t_white={VERY_THIN_T_WHITE}')
    print(f'    İnce               : non_bg < %{THIN_THRESHOLD*100:.0f}  → t_white={THIN_T_WHITE} (veya vadi)')
    print(f'    Normal             : non_bg ≥ %{THIN_THRESHOLD*100:.0f}  → histogram vadi')
    print()
    print('  [YENİLİK 4] Geri Besleme Döngüsü:')
    print(f'    Tetik              : shave/brain > %{SAFETY_RATIO*100:.0f}')
    print(f'    Adım               : t_white += {FEEDBACK_STEP} / iterasyon')
    print(f'    Maksimum iterasyon : {MAX_FEEDBACK_ITER}')
    print()

    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    clean_dir  = output_dir / 'cleaned_png'
    mask_dir   = output_dir / 'halo_mask_png'
    clean_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob('*.png'))
    print(f'  📁 Bulunan PNG     : {len(png_files)}')
    if not png_files:
        print('  ❌ Dosya bulunamadı! INPUT_DIR\'i kontrol edin.')
        sys.exit(1)

    n_workers = max(1, mp.cpu_count() - 1)
    print(f'  ⚙️  CPU çekirdeği  : {n_workers} / {mp.cpu_count()}')
    print()

    worker = partial(process_file, clean_dir=clean_dir, mask_dir=mask_dir)

    rows, done = [], 0
    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(worker, png_files, chunksize=4):
            rows.append(result)
            done += 1
            if done % 100 == 0 or done == len(png_files):
                pct = done / len(png_files) * 100
                print(f'  ⏳ {done}/{len(png_files)}  ({pct:.1f}%) tamamlandı...')

    df = pd.DataFrame(rows).sort_values('filename').reset_index(drop=True)

    # ── Özet istatistikler ────────────────────────────────────────────────────
    errors        = df[df['error'] != '']
    very_thin_cnt = (df['slice_class'] == 'very_thin').sum()
    thin_cnt      = (df['slice_class'] == 'thin').sum()
    normal_cnt    = (df['slice_class'] == 'normal').sum()
    valley_cnt    = df['valley_found'].sum()
    feedback_cnt  = df['feedback_triggered'].sum()
    high_conf     = (df['confidence'] == 'high').sum()
    med_conf      = (df['confidence'] == 'medium').sum()
    low_conf      = (df['confidence'] == 'low').sum()

    print()
    print('=' * 68)
    print('  ✅  TAMAMLANDI')
    print('=' * 68)
    print(f'  Hata               : {len(errors)}')
    print()
    print('  Dilim Sınıfları:')
    print(f'    Normal  (≥%{THIN_THRESHOLD*100:.0f})       : {normal_cnt}   ({normal_cnt/len(df)*100:.1f}%)')
    print(f'    İnce    (%{VERY_THIN_THRESHOLD*100:.0f}–%{THIN_THRESHOLD*100:.0f})   : {thin_cnt}   ({thin_cnt/len(df)*100:.1f}%)')
    print(f'    Çok İnce(<%{VERY_THIN_THRESHOLD*100:.0f})       : {very_thin_cnt}   ({very_thin_cnt/len(df)*100:.1f}%)')
    print()
    print('  Histogram Vadi:')
    print(f'    Vadi bulundu       : {valley_cnt}   ({valley_cnt/len(df)*100:.1f}%)')
    print(f'    Yedek kullanıldı   : {len(df)-valley_cnt}   ({(len(df)-valley_cnt)/len(df)*100:.1f}%)')
    print()
    print('  Geri Besleme:')
    print(f'    Tetiklendi         : {feedback_cnt}   ({feedback_cnt/len(df)*100:.1f}%)')
    print(f'    Tetiklenmedi       : {len(df)-feedback_cnt}   ({(len(df)-feedback_cnt)/len(df)*100:.1f}%)')
    print()
    print('  Güven Skoru:')
    print(f'    Yüksek             : {high_conf}   ({high_conf/len(df)*100:.1f}%)')
    print(f'    Orta               : {med_conf}   ({med_conf/len(df)*100:.1f}%)')
    print(f'    Düşük              : {low_conf}   ({low_conf/len(df)*100:.1f}%)')
    print()
    print(f'  t_white ort.         : {df["t_white_final"].mean():.1f}  (v2\'de sabit 235)')
    print(f'  t_white aralığı      : [{df["t_white_final"].min()}, {df["t_white_final"].max()}]')
    print(f'  max_depth ort.       : {df["max_depth_used"].mean():.1f}  (v2\'de sabit 25)')
    print(f'  Ort. halo            : {df["halo_pixels_removed"].mean():.0f} px / dilim')

    if len(errors):
        print()
        print('  ⚠️  Hatalı dosyalar:')
        print(errors[['filename', 'error']].to_string(index=False))

    # ── Excel kaydet ──────────────────────────────────────────────────────────
    excel_path = output_dir / 'traslama_v3_istatistik.xlsx'

    # Kolon sırası: okunabilirlik için grupla
    col_order = [
        'filename', 'error',
        'height', 'width', 'total_pixels',
        'img_min', 'img_max', 'img_mean', 'img_std', 'img_p95',
        'slice_class', 'non_bg_pct', 'non_bg_pixels',
        't_bg', 't_white_init', 't_white_final', 'valley_found',
        'max_depth_used',
        'feedback_triggered', 'n_iterations', 'final_shave_ratio', 'confidence',
        'halo_pixels_removed', 'halo_pct',
        'dir_left_px', 'dir_right_px', 'dir_top_px', 'dir_bottom_px',
        'dir_dominant', 'dir_imbalance',
    ]
    existing_cols = [c for c in col_order if c in df.columns]
    df[existing_cols].to_excel(excel_path, index=False)

    print()
    print(f'  💾 Excel   : {excel_path}')
    print(f'  📁 Cleaned : {clean_dir}')
    print(f'  📁 Mask    : {mask_dir}')
    print('=' * 68)