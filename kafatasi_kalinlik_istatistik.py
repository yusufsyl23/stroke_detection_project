"""
Kafatası Kalınlık ve Histogram Peak Analizi
============================================
Tıraşlanmamış (orijinal) beyin CT görüntüleri üzerinde çalışır.

Her görüntü için hesaplananlar:
  - Gerçek kafatası kalınlığı (8 yönde ray bazlı ölçüm)
  - Histogram peak pozisyonları (beyin peak, kafatası peak)
  - İki peak arası mesafe (GMM / peak-tespiti için kritik)
  - fg_pct (dilim büyüklüğü)
  - Önerilen t_white (peak-arası minimum yöntemi)
  - Vadi yöntemi t_white (karşılaştırma için)

KULLANIM:
    python kafatasi_kalinlik_analizi.py

YOLLAR (aşağıda ayarla):
    INPUT_DIR  : Tıraşlanmamış beyin PNG'lerinin klasörü
    OUTPUT_XLSX: Çıktı Excel dosyası
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# AYARLAR — kendi yollarınla değiştir
# ============================================================
INPUT_DIR   = r"C:\Users\Yusuf Soylu\Desktop\Beyin Traslama\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V3\Temiz_Beyin_v3_EREN\cleaned_png"
OUTPUT_XLSX = r"C:\Users\Yusuf Soylu\Desktop\kafatasi_kalinlik_istatistik_v3.xlsx"
# ============================================================

# Piksel yoğunluk bantları (histogram analizi için)
BAND_DARK_MAX   = 50    # arka plan / karanlık piksel üst sınırı
BAND_BRAIN_MIN  = 51    # beyin dokusu başlangıcı
BAND_BRAIN_MAX  = 180   # beyin dokusu bitişi
BAND_MID_MIN    = 181   # geçiş bölgesi başlangıcı
BAND_MID_MAX    = 219   # geçiş bölgesi bitişi
BAND_WHITE_MIN  = 220   # kafatası kemiği başlangıcı


def get_t_bg(img: np.ndarray, bw: int = 5) -> int:
    """Kenar piksellerinden arka plan eşiği hesapla. V1 den beri aynı"""
    border = np.concatenate([
        img[:bw, :].ravel(), img[-bw:, :].ravel(),
        img[:, :bw].ravel(), img[:, -bw:].ravel()
    ])
    return int(np.percentile(border, 99.9))

# V4'ün MAX_DEPTH tablosu buradan türedi.
def measure_skull_thickness_rays(img: np.ndarray, t_bg: int, t_white: int,
                                  n_angles: int = 36) -> dict:
    """
    Nesnenin ağırlık merkezinden dışa doğru ışın yayarak
    kafatası kalınlığını ölçer.

    Her ışın:
      - Ön plan sınırından içe girer
      - İlk beyaz pixel'den (>= t_white) beyaz olmayan pixel'e kadar ölçer
      - Bu mesafe = kafatası kalınlığı o yönde

    Returns dict ile: mean, median, max, min, p25, p75, p90, p95,
                       n_rays_measured, thickness_values listesi
    """
    h, w = img.shape
    fg_mask = img > t_bg
    if fg_mask.sum() == 0:
        return _empty_ray_result()

    # Ağırlık merkezi
    # ys → True olan hücrelerin satır numaraları
    # xs → True olan hücrelerin sütun numaraları
    ys, xs = np.where(fg_mask)
    # Önplan maskesindeki tüm True piksellerin koordinat ortalaması = beynin merkezi. Buradan ışınlar gönderilecek.
    cy, cx = int(ys.mean()), int(xs.mean())

    thicknesses = []
    # # 0°, 10°, 20°, ..., 350° — her 10 derecede bir
    # Kafatası bir halka. Merkezden 36 farklı açıda ışın gönder, her yönde "beyaz bloğun kalınlığı" ne kadar?
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    for angle in angles:
        dy, dx = np.sin(angle), np.cos(angle)

        # 1. Ön plan sınırını bul: merkezden dışa giderek fg bittiği nokta
        fg_end_r = None     # henüz sınır bulunamadı
        for r in range(1, max(h, w)):       # 1'den 511'e kadar
            ny = int(cy + r * dy)           # = int(256 + r × 0) = 256
            nx = int(cx + r * dx)           # = int(256 + r × 1) = 256 + r

            # "Görüntü sınırına çarpmadan önce dur, son geçerli pikseli kaydet." Olmasa program IndexError ile çökerdi.
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                fg_end_r = r - 1
                break

            # İlerlerken artık ön plan dışına çıktık yani fg_mask[ny, nx] = False oldu. Bir önceki True değerinin r değerini tut.
            if not fg_mask[ny, nx]:
                fg_end_r = r - 1
                break

        if fg_end_r is None or fg_end_r < 2:
            continue

        # 2. Dış kenardan içe doğru beyaz bloğu ölç
        white_start = None
        white_end = None
        # Dıştan içe: fg_end_r → 1
        for r in range(fg_end_r, 0, -1):
            ny, nx = int(cy + r * dy), int(cx + r * dx)
            if ny < 0 or ny >= h or nx < 0 or nx >= w:    # Sınıra çarparsan atla
                continue
            val = img[ny, nx]
            if val >= t_white:          # İlk >= t_white piksel → kafatası başladı (white_start)
                if white_start is None:
                    white_start = r
            else:
                if white_start is not None:     # Beyazdan düşük değere geçince → kafatası bitti (white_end)
                    white_end = r
                    break

        if white_start is not None:
            if white_end is None:
                white_end = 0  # beyaz merkezine kadar devam etti
            thickness = white_start - white_end     # Kalınlık = white_start - white_end
            if thickness > 0:
                thicknesses.append(thickness)

    if not thicknesses:
        return _empty_ray_result()

    # 36 yönden gelen kalınlık değerleri toplanır
    t = np.array(thicknesses)
    return {
        'skull_thickness_mean':   float(t.mean()),
        'skull_thickness_median': float(np.median(t)),
        'skull_thickness_max':    float(t.max()),
        'skull_thickness_min':    float(t.min()),
        'skull_thickness_p25':    float(np.percentile(t, 25)),
        'skull_thickness_p75':    float(np.percentile(t, 75)),
        'skull_thickness_p90':    float(np.percentile(t, 90)),
        'skull_thickness_p95':    float(np.percentile(t, 95)),
        'n_rays_measured':        len(thicknesses),
    }


def _empty_ray_result() -> dict:
    return {k: 0.0 for k in [
        'skull_thickness_mean', 'skull_thickness_median',
        'skull_thickness_max', 'skull_thickness_min',
        'skull_thickness_p25', 'skull_thickness_p75',
        'skull_thickness_p90', 'skull_thickness_p95', 'n_rays_measured'
    ]}


def find_peaks_twhite(fg_pixels: np.ndarray, img_max: int) -> dict:
    """
    Histogram üzerinde robust iki-peak tespiti ile t_white belirle.
    Sabit pencere yok — tüm histogram taranır.

    Yöntem:
      1. Smoothed histogram → find_peaks
      2. En yüksek 2 peak seç (minimum 40 HU aralıkla)
      3. İkisi arasındaki minimum = t_white
      4. Fallback koşulları belgelenmiş şekilde döner
    """
    # 10'dan az önplan pikseli → histogram çizmeye değmez, anlamsız sonuç çıkar.
    # Direkt fallback. Pratikte very_thin slice'larda bile binlerce piksel var, bu koşul neredeyse hiç tetiklenmez.
    if len(fg_pixels) < 10:
        return _peak_fallback(img_max, 'insufficient_pixels')


    """
    np.histogram: 256 kutu, 0–255 aralığı. Her kutu 1 HU genişliğinde. hist[0] → değeri 0 olan piksel sayısı. hist[120] → değeri 120 olan piksel sayısı.
    edges: Kutu sınırları. 256 kutu için 257 sınır değeri:
    edges = [0.0, 1.0, 2.0, ..., 255.0, 256.0]
    centers: Her kutunun merkez değeri.
    edges[:-1] → son eleman hariç, ilk 256 eleman . edges[1:] → ilk eleman hariç, son 256 eleman
    
    centers[0]   = (edges[0]   + edges[1])   / 2 = (0.0     + 0.996)   / 2 = 0.498
    centers[1]   = (edges[1]   + edges[2])   / 2 = (0.996   + 1.992)   / 2 = 1.494
    centers[2]   = (edges[2]   + edges[3])   / 2 = (1.992   + 2.988)   / 2 = 2.490
    
    Amacı Ne?
    Her bin bir aralığı temsil eder:
    Bin 0:   0.0   → 0.996  arası pikseller burada sayılır
    Bin 1:   0.996 → 1.992  arası pikseller burada sayılır
    Bin 120: 119.5 → 120.5  arası pikseller burada sayılır
    
    Sonunda t_white_peak = int(centers[valley_idx]) diyoruz. centers[120] = 119.94 → int → 119.
    Amaç: Bin indeksini (0, 1, 2... 255) gerçek HU değerine (0.5, 1.5, 2.5... 254.5) çevirmek.
    """
    hist, edges = np.histogram(fg_pixels, bins=256, range=(0, 255))
    centers = (edges[:-1] + edges[1:]) / 2

    # Savitzky Golay Smooth
    """
       Savitzky-Golay için pencere boyutu. İki kural:

       Kural 1: min(11, len(hist) // 10 * 2 + 1)
       len(hist) = 256 her zaman. 256 // 10 = 25. 25 × 2 + 1 = 51. min(11, 51) = 11. Yani bu veri setinde her zaman 11 çıkıyor. 
       Ama fonksiyon genel amaçlı yazılmış — farklı bins değeriyle çağrılırsa farklı pencere seçilir.

       Kural 2: max(5, window if window % 2 == 1 else window + 1)
       Savitzky-Golay tek sayı pencere ister. window % 2 == 1 → tek sayıysa olduğu gibi kullan. 
       Çift sayıysa +1 yap. Sonra max(5, ...) → minimum 5. Bu veri setinde: 11 % 2 = 1 → tek → max(5, 11) = 11.
    """
    # Bu satır şunu söylüyor: "histogram uzunluğunun %20'si kadar pencere al, ama en fazla 11 olsun." 256 uzunluklu histogramda
    # maksimum 11 mantıklı — çok büyük pencere gerçek pikleri ezer.
    window = min(11, len(hist) // 10 * 2 + 1)   # min(11, 51) = 11
    window = max(5, window if window % 2 == 1 else window + 1)  # max(5,11) = 11      # Tek sayı garantisi
    # Bu veri setinde her zaman window=11 çıkıyor.

    """
    Savitzky-Golay Nasıl Çalışıyor?

    Kayan Ortalama Ne Yapıyor?
    Her noktayı komşularının aritmetik ortalaması ile değiştiriyor. Tüm komşulara eşit ağırlık veriliyor.

    Savitzky-Golay Ne Yapıyor?
    Her pencere için komşulara eşit ağırlık vermek yerine o komşulara bir polinom fit ediyor ve polinomun merkez değerini kullanıyor.

    Somut adımlar:
    Pencere = [10, 12, 200, 11, 9], merkez = 200:
    x konumları: [-2, -1, 0, 1, 2]
    y değerleri: [10, 12, 200, 11, 9]

    Bu 5 noktaya 2. derece polinom fit et:
    y = ax² + bx + c

    Least squares ile a, b, c bul. Polinom şöyle çıksın:
    y = -40x² + 5x + 195

    Merkezdeki tahmin (x=0):
    y(0) = -40(0) + 5(0) + 195 = 195
    
    Ham:    [10, 12, 200, 11, 9]   ← ortada ani tepe
    Window=3 kayan ortalama:
      [10, 12, 200] / 3 = 74   ← tepe 200'den 74'e düştü
      [12, 200, 11] / 3 = 74
      [200, 11,  9] / 3 = 73
    Sonuç: [10, 74, 74, 73, 9]  ← tepe tamamen ezildi
    
    Ham:    [10, 12, 200, 11, 9]
    SG window=5, polyorder=2:
    Sonuç: [11, 13, 198, 12, 10]  ← tepe 200 → 198, neredeyse korundu

    Ham değer 200'dü → polinom tahmini 195. Kayan ortalama 74 veriyordu. Savitzky-Golay çok daha yakın.

    Neden Polinom Bu Kadar İyi?
    Gerçek pik zaten polinom şeklinde — yükselip alçalıyor. Polinom fit bu yapıyı tanıyor ve koruyor. Kayan ortalama körü körüne ortaliyor, yapıyı görmüyor.
    """

    try:
        """
        savgol_filter(data, window_length=11, polyorder=2): Her bin için 5 soldan 5 sağa 11 komşu al, 
        bunlara 2. derece polinom fit et, merkez değeri hesapla. Tüm 256 bin için bunu tekrarla.
        """
        hist_s = savgol_filter(hist.astype(float), window_length=window, polyorder=2)
        # Savgol bazen negatif değer üretir, özellikle histogramın uçlarında (0 veya 255 civarı). Negatif piksel sayısı anlamsız → sıfırla.
        hist_s = np.maximum(hist_s, 0)
    except Exception:
        # Hata alırsak basit 7 pencereli kayan ortalama uygula
        hist_s = np.convolve(hist.astype(float), np.ones(7)/7, mode='same')

    """
    512×512 görüntü, önplan ~50.000 piksel. 
    
    Ham histogram (düzleştirilmemiş):                                       Savgol Düzleştirme Sonrası hist_s
    
    HU Değeri | Piksel Sayısı | Açıklama                             HU Değeri | hist_s        | Açıklama                    
    ----------|---------------|----------                            ----------|---------------|----------
        60    |     200       | az beyin kenarı                          60    |     350       |                                                               
        70    |     800       |                                          70    |    1100       |
        80    |    2100       |                                          80    |    2800       |        
        90    |    4500       |                                          90    |    5200       |                            
       100    |    7800       |                                         100    |    9000       |
       110    |   12000       |                                         110    |   13500       |
       115    |    8000       | ← ANİ DİP (gürültü)                     115    |   15200       | ← Ani dip yumuşadı
       120    |   18000       | GERÇEK BEYİN TEPESİ                     120    |   17800       | BEYİN TEPESİ (korundu)
       125    |   15000       |                                         125    |   14500       |
       130    |    9000       |                                         130    |    8500       |
       140    |    3000       |                                         140    |    2800       |
       150    |    1200       |                                         150    |    1300       |
       155    |    1800       | ← KÜÇÜK SAHTE PİK (artefakt)            155    |    1400       | ← Küçük sahte pik düzleşti
       160    |    1100       |                                         160    |    1050       |
       170    |     400       |                                         170    |     450       |
       175    |     600       | ← KÜÇÜK SAHTE PİK (gürültü)             175    |     520       | ← Küçük sahte pik düzleşti
       180    |     300       |                                         180    |     320       |
       190    |     150       | vadi bölgesi                            190    |     160       |
       200    |      80       | ← gerçek vadi                           200    |      90       | vadi
       210    |     200       |                                         210    |     220       |
       220    |    1500       |                                         220    |    1600       |
       230    |    4000       |                                         230    |    4200       |
       240    |    7500       | GERÇEK KAFATASI TEPESİ                  240    |    7300       | KAFATASI TEPESİ (korundu)
       250    |    5000       |                                         250    |    4800       |
       255    |    2000       |                                         255    |    1900       |
    """

    # Peak bul
    """
    Histogramın en yüksek noktasının %3'ünden kısa tepeleri yok say. Neden? Gürültü ve küçük artefaktlar çok sayıda sahte pik üretir. 
    Bunları eleyerek sadece anlamlı yapısal tepeleri al.
    
    height=min_height → minimum yükseklik filtresi
    distance=20 → iki pik arası minimum 20 bin (20 HU)
    
    distance=20 → iki pik arası minimum 20 HU. find_peaks pikleri yüksekliğe göre sıralar, yakındaki zayıfı eler:
    Pik 120 (17800) → en güçlü, kalır
    Pik 155 (1400)  → 120'ye mesafesi = 35 ≥ 20 → kalır
    Pik 240 (7300)  → 155'e mesafesi = 85 ≥ 20 → kalır
    Pik 175 (520)   → height < 534 → zaten elendi
    
    props: Her pikin özellikleri (yükseklik, prominence vs). Bu fonksiyonda kullanılmıyor ama find_peaks her zaman iki değer döndürür, 
    değişkene atanması gerekiyor.
    """
    min_height = hist_s.max() * 0.03
    # peak_indices = histogram'ın kaçıncı bininde pik var.
    peak_indices, props = find_peaks(hist_s, height=min_height, distance=20)

    # # Döngüde hiçbir pik 20 HU uzakta bulunamadı → tüm pikler birbirine yakın → gerçek iki doku ayrımı yok → fallback.
    if len(peak_indices) < 2:
        return _peak_fallback(img_max, 'less_than_2_peaks')

    # En yüksek 2 peak'i seç, ama aralarında en az 40 HU olsun
    peak_heights = hist_s[peak_indices]     # Bulunan her pikin yüksekliği
    # np.argsort(peak_heights): peak_heights = [17800, 1400, 7300] bu diziyi küçükten büyüğe sıralasam, elemanlar hangi sırayla gelirdi?
    # np.argsort(peak_heights) = [1, 2, 0]
    # Amaç: pikleri yüksekliklerine göre büyükten küçüğe sırala, ama konum değerlerini koru.
    sorted_peaks = peak_indices[np.argsort(peak_heights)[::-1]]

    """
    ADIM                  | Değer
    ----------------------|---------------------------
    peak_indices          | [120,   155,   240  ]
    peak_heights          | [17800, 1400,  7300 ]
                          |   [0]    [1]    [2]   ← indeksler
                          |
    argsort (küçük→büyük) | [1,     2,     0    ]
                          | (1400) (7300) (17800)
                          |
    [::-1] (büyük→küçük)  | [0,     2,     1    ]
                          | (17800)(7300) (1400)
                          |
    peak_indices[...] =   | [120,   240,   155  ]
    sorted_peaks          |  ^^^     ^^^    ^^^
                          | 1.inci  2.inci  3.üncü yüksek
    """

    selected = [sorted_peaks[0]]        # en yüksek pik
    # Döngü mantığı: En yüksek pikten (120) en az 40 HU uzakta olan bir sonraki en yüksek piki bul.
    """
    selected = [120, 240]. Aradaki 180 ve 200 atlandı — en yüksek iki pik bunlar değildi.
    Neden 40 HU? Kafatası analiz scripti bu veri setinde iki gerçek tepe arasında minimum 40 HU olduğunu gözlemledi. 
    40'tan yakın iki pik aynı dokunun alt bantları.
    """
    for p in sorted_peaks[1:]:
        if abs(p - selected[0]) >= 40:
            selected.append(p)
            break

    # Döngüde hiçbir pik 40 HU uzakta bulunamadı → tüm pikler birbirine yakın → gerçek iki doku ayrımı yok → fallback.
    if len(selected) < 2:
        return _peak_fallback(img_max, 'peaks_too_close')

    # sorted(selected): İki elemanı küçükten büyüğe sırala.
    # Konum olarak düşük olan (low_peak) = düşük HU = beyin. Konum olarak yüksek olan (high_peak) = yüksek HU = kafatası.
    low_peak, high_peak = sorted(selected)
    # İki vadi arasındaki uzaklığı ölçer. 40 dan büyükse t_white güvenilir değilse FALLBACK olur.
    """
    Beyin tepesi: 120        Kafatası tepesi: 240
         ▲                          ▲
         |                          |
    ─────┼──────────────────────────┼─────
       120        180-200          240
                  (vadi)
    
    → İki doku net ayrılmış
    → Vadi temiz, t_white güvenilir
    → separation=120 >= 40 → GEÇER
    """
    peak_separation = int(high_peak - low_peak)

    """
    selected    = [120, 240]
    sorted      = [120, 240]
    low_peak    = 120  ← beyin tepesi (düşük HU)
    high_peak   = 240  ← kafatası tepesi (yüksek HU)
    separation  = 120
    """

    # İki peak arası minimum = t_white bulma
    valley_region = hist_s[low_peak:high_peak + 1]  # hist_s[120:241]: İki pik arasındaki bölge. 121 elemanlı alt dizi. VADİ
    valley_offset = int(np.argmin(valley_region))   # Bu bölgedeki minimum (dip) noktanın konumu
    valley_idx = low_peak + valley_offset
    t_white_peak = int(centers[valley_idx])

    # Vadi yöntemi karşılaştırma (V3 mevcut yöntemi — 150-230 sabit pencere)
    lo_old, hi_old = 150, 230
    region_old = hist_s[lo_old:hi_old + 1]
    valley_old_offset = int(np.argmin(region_old))
    valley_old = lo_old + valley_old_offset
    valley_old_val = float(hist_s[valley_old])
    left_max_old = float(hist_s[:valley_old].max()) if valley_old > 0 else 1.0
    right_max_old = float(hist_s[valley_old + 1:].max()) if valley_old < 255 else 1.0
    is_real_old = (valley_old_val < 0.95 * left_max_old and
                   valley_old_val < 0.95 * right_max_old and
                   left_max_old > 10 and right_max_old > 10)
    t_white_v3_old = valley_old if is_real_old else max(225, int(0.92 * img_max))

    return {
        'brain_peak_pos': int(centers[low_peak]),           # beyin tepesi HU konumu
        'skull_peak_pos': int(centers[high_peak]),          # kafatası tepesi HU konumu
        'peak_separation_hu': peak_separation,              # iki tepe arası mesafe
        'valley_pos': t_white_peak,                         # vadi konumu = t_white
        't_white_new': t_white_peak,                        # yeni yöntem t_white
        't_white_v3_old': t_white_v3_old,                   # eski yöntem t_white
        't_white_diff': t_white_peak - t_white_v3_old,      # fark
        'brain_peak_height': float(hist_s[low_peak]),       # beyin tepesi yüksekliği
        'skull_peak_height': float(hist_s[high_peak]),      # kafatası tepesi yüksekliği
        'valley_depth_ratio': float(hist_s[valley_idx]) / max(hist_s[low_peak], 1),
        'n_peaks_found': len(peak_indices),                 # toplam pik sayısı
        'peak_method': 'two_peak_valley',                   # başarılı
        'fallback_reason': '',                              # boş = başarılı
    }


def _peak_fallback(img_max: int, reason: str) -> dict:
    """
    Senaryo 1 — Very thin slice:
        fg_pixels az → histogram iki net tepe oluşturamıyor
        → less_than_2_peaks → fallback

    Senaryo 2 — Kafatası çok ince:
        Kafatası tepesi çok küçük, beyin tepesiyle 40 HU'dan yakın
        → peaks_too_close → fallback

    Senaryo 3 — img_max düşük hasta:
        fg_pixels < 10 (neredeyse hiç önplan yok)
        → insufficient_pixels → fallback
    """

    # V2'nin sabit formülü
    t_white = max(225, int(0.92 * img_max))
    return {
        'brain_peak_pos':       0,
        'skull_peak_pos':       0,
        'peak_separation_hu':   0,          # "bu slice'ta pik tespiti başarısız" demek
        'valley_pos':           t_white,
        't_white_new':          t_white,
        't_white_v3_old':       t_white,
        't_white_diff':         0,
        'brain_peak_height':    0.0,
        'skull_peak_height':    0.0,
        'valley_depth_ratio':   0.0,
        'n_peaks_found':        0,          # "hiç pik bulunamadı" demek
        'peak_method':          'fallback',
        'fallback_reason':      reason,
    }


def analyze_image(fpath: Path) -> dict:
    """Tek görüntü için tüm metrikleri hesapla."""
    try:
        img = np.array(Image.open(fpath).convert('L'), dtype=np.uint8)
    except Exception as e:
        return {'filename': fpath.name, 'error': str(e)}

    h, w = img.shape
    total_px = h * w
    img_max = int(img.max())
    img_min = int(img.min())

    # t_bg
    t_bg = get_t_bg(img)
    fg_mask = img > t_bg
    fg_px = int(fg_mask.sum())
    fg_pct = fg_px / total_px * 100

    if fg_px == 0:      # Boş dilim kontrolü
        return {
            'filename': fpath.name, 'patient_id': _get_patient_id(fpath.name),
            'slice_no': _get_slice_no(fpath.name), 'img_max': img_max,
            'fg_px': 0, 'fg_pct': 0.0, 'slice_class': 'empty',
            'error': 'no_foreground'
        }

    fg_pixels = img[fg_mask]
    non_bg_pct = fg_px / total_px

    # Dilim sınıfı (V3 mantığı)
    if non_bg_pct < 0.02:
        slice_class = 'very_thin'
    elif non_bg_pct < 0.08:
        slice_class = 'thin'
    else:
        slice_class = 'normal'

    # Histogram band analizi
    hist_dark_pct  = float((fg_pixels <= BAND_DARK_MAX).mean() * 100)
    hist_brain_pct = float(((fg_pixels >= BAND_BRAIN_MIN) & (fg_pixels <= BAND_BRAIN_MAX)).mean() * 100)
    hist_mid_pct   = float(((fg_pixels >= BAND_MID_MIN) & (fg_pixels <= BAND_MID_MAX)).mean() * 100)
    hist_white_pct = float((fg_pixels >= BAND_WHITE_MIN).mean() * 100)

    # Peak tespiti ve t_white (yeni yöntem)
    peak_result = find_peaks_twhite(fg_pixels, img_max)

    # V3 mevcut dinamik derinlik
    raw_depth = int(np.sqrt(max(fg_px, 1)) * 0.06)
    v3_max_depth = int(np.clip(raw_depth, 8, 20))

    # Gerçek kafatası kalınlığı (t_white_new kullan)
    t_white_for_skull = peak_result['t_white_new']
    skull_result = measure_skull_thickness_rays(
        img, t_bg, t_white_for_skull, n_angles=36
    )

    # Tıraşlanabilir piksel sayısı her iki yöntemle
    tirable_new = int((img >= t_white_for_skull).sum())
    tirable_old = int((img >= peak_result['t_white_v3_old']).sum())

    row = {
        'filename':     fpath.name,
        'patient_id':   _get_patient_id(fpath.name),
        'slice_no':     _get_slice_no(fpath.name),
        'img_max':      img_max,
        'img_min':      img_min,
        'total_px':     total_px,
        'fg_px':        fg_px,
        'fg_pct':       round(fg_pct, 4),
        'slice_class':  slice_class,
        # Histogram bantları
        'hist_dark_pct':  round(hist_dark_pct, 2),
        'hist_brain_pct': round(hist_brain_pct, 2),
        'hist_mid_pct':   round(hist_mid_pct, 2),
        'hist_white_pct': round(hist_white_pct, 2),
        # V3 dinamik derinlik
        'v3_raw_depth':  raw_depth,
        'v3_max_depth':  v3_max_depth,
        # Peak tespiti sonuçları
        **peak_result,
        # Tıraşlanabilir piksel
        'tirable_px_new': tirable_new,
        'tirable_px_old': tirable_old,
        # Gerçek kafatası kalınlığı (ışın ölçümü)
        **skull_result,
        'error': '',
    }
    return row


def _get_patient_id(fname: str) -> str:
    """'02_slice0064.png' → '02'"""
    try:
        return fname.split('_')[0]
    except Exception:
        return ''


def _get_slice_no(fname: str) -> int:
    """'02_slice0064.png' → 64"""
    try:
        part = fname.split('slice')[1].split('.')[0]
        return int(part)
    except Exception:
        return -1


def main():
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"HATA: Klasör bulunamadı: {INPUT_DIR}")
        return

    png_files = sorted(input_path.glob('*.png'))
    if not png_files:
        png_files = sorted(input_path.rglob('*.png'))

    print(f"Toplam {len(png_files)} PNG bulundu.")
    print(f"İşleniyor...")

    rows = []
    for i, fpath in enumerate(png_files):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(png_files)} — {fpath.name}")
        row = analyze_image(fpath)
        rows.append(row)

    df = pd.DataFrame(rows)

    # -------------------------------------------------------
    # Özet istatistikler
    # -------------------------------------------------------
    valid = df[df.error == ''].copy()

    summary_rows = []
    for patient_id in sorted(valid.patient_id.unique()):
        p = valid[valid.patient_id == patient_id]
        summary_rows.append({
            'patient_id':               patient_id,
            'n_slices':                 len(p),
            'skull_thickness_mean_avg': round(p.skull_thickness_mean.mean(), 1),
            'skull_thickness_max_avg':  round(p.skull_thickness_max.mean(), 1),
            'skull_thickness_max_max':  round(p.skull_thickness_max.max(), 1),
            'peak_separation_avg':      round(p.peak_separation_hu.mean(), 1),
            'fallback_count':           (p.peak_method == 'fallback').sum(),
            'v3_depth_vs_skull_ratio':  round(
                (p.v3_max_depth / p.skull_thickness_mean.replace(0, np.nan)).mean(), 2
            ),
        })
    summary_df = pd.DataFrame(summary_rows)

    # -------------------------------------------------------
    # Dilim sınıfı bazlı özet
    # -------------------------------------------------------
    slice_summary = valid.groupby('slice_class').agg(
        n=('filename', 'count'),
        skull_mean_avg=('skull_thickness_mean', 'mean'),
        skull_max_avg=('skull_thickness_max', 'mean'),
        skull_max_p90=('skull_thickness_max', lambda x: np.percentile(x, 90)),
        peak_sep_avg=('peak_separation_hu', 'mean'),
        t_white_new_avg=('t_white_new', 'mean'),
        t_white_old_avg=('t_white_v3_old', 'mean'),
        v3_depth_avg=('v3_max_depth', 'mean'),
        fallback_pct=('peak_method', lambda x: (x == 'fallback').mean() * 100),
    ).round(2)

    # -------------------------------------------------------
    # Yetersiz derinlik analizi
    # (v3_max_depth < skull_thickness_mean → V3 kafatası bitirmeden duruyor)
    # -------------------------------------------------------
    valid['depth_gap'] = valid.skull_thickness_mean - valid.v3_max_depth
    valid['depth_insufficient'] = valid.depth_gap > 0
    insuf = valid[valid.depth_insufficient].copy()
    print(f"\nV3 derinliği yetersiz olan dilimler: {len(insuf)} / {len(valid)} ({len(insuf)/len(valid)*100:.1f}%)")
    print(f"  Ortalama eksik derinlik: {insuf.depth_gap.mean():.1f}px")
    print(f"  Maksimum eksik derinlik: {insuf.depth_gap.max():.1f}px")

    # Fallback oranı
    fallback = valid[valid.peak_method == 'fallback']
    print(f"\nPeak tespiti fallback'e düşen: {len(fallback)} / {len(valid)} ({len(fallback)/len(valid)*100:.1f}%)")
    print(f"  Fallback nedenleri:\n{fallback.fallback_reason.value_counts().to_string()}")

    # t_white farkı
    print(f"\nt_white fark istatistiği (yeni - eski):")
    print(f"  Ort: {valid.t_white_diff.mean():.1f}  Med: {valid.t_white_diff.median():.1f}")
    print(f"  >+10 (yeni daha yüksek): {(valid.t_white_diff > 10).sum()}")
    print(f"  <-10 (yeni daha düşük):  {(valid.t_white_diff < -10).sum()}")

    # -------------------------------------------------------
    # Excel'e yaz
    # -------------------------------------------------------
    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        valid.to_excel(writer, sheet_name='Tüm Dilimler', index=False)
        summary_df.to_excel(writer, sheet_name='Hasta Özeti', index=False)
        slice_summary.to_excel(writer, sheet_name='Dilim Sınıfı Özeti', index=False)

        # En sorunlu dilimler (depth_gap en büyük)
        top_gap = valid.nlargest(50, 'depth_gap')[
            ['filename', 'patient_id', 'slice_no', 'fg_pct', 'slice_class',
             'skull_thickness_mean', 'skull_thickness_max', 'v3_max_depth',
             'depth_gap', 'peak_separation_hu', 't_white_new', 't_white_v3_old']
        ]
        top_gap.to_excel(writer, sheet_name='En Büyük Derinlik Açığı', index=False)

        # Fallback dilimler
        if len(fallback) > 0:
            fallback[['filename', 'patient_id', 'slice_no', 'fg_pct', 'slice_class',
                       'img_max', 'fallback_reason', 't_white_new', 't_white_v3_old']
                     ].to_excel(writer, sheet_name='Fallback Dilimler', index=False)

    print(f"\nExcel kaydedildi: {OUTPUT_XLSX}")
    print("Tamamlandı.")


if __name__ == '__main__':
    main()