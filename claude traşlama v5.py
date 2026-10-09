# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              TRAŞLAMA ALGORİTMASI — SÜRÜM 5 (traslamaV5.py)               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  V4.1 → V5 DEĞİŞİKLİKLERİ:                                                 ║
║                                                                              ║
║  DEĞİŞİKLİK 1 — PİK SEÇİM MANTIĞI (TEMEL DÜZELTME)                        ║
║    V4.x: "En büyük 2 pik" → beynin 2 alt bandı (gri+beyaz madde)           ║
║           seçiliyordu, gerçek kafatası piki atlanıyordu.                    ║
║    V5:    Histogram KONUMA GÖRE 2 bölgeye ayrılır:                          ║
║           brain_zone:  51-179 HU  → en yüksek pik = beyin                  ║
║           skull_zone: 180-255 HU  → en yüksek pik = kafatası               ║
║           Kafatası piki bulunamazsa fallback.                               ║
║                                                                              ║
║  DEĞİŞİKLİK 2 — PEAK_MIN_SEPARATION_HU KALDIRILDI                          ║
║    Konum bazlı seçimde separation her zaman >= 180-179 = 1.                 ║
║    Gerçekte 180+ kafatası piki ile 51-179 beyin piki arasında               ║
║    separation her zaman anlamlı. Kontrol gereksiz.                          ║
║                                                                              ║
║  DEĞİŞİKLİK 3 — img_p95 TABANLI GÜVENLİK SINIRI                           ║
║    t_white = max(t_white, img_p95_fg + SAFETY_MARGIN)                      ║
║    Silme eşiği hiçbir zaman beyin dokusunun %95 yoğunluğuna                ║
║    SAFETY_MARGIN'dan yakın olamaz. Beyin güvencesi.                        ║
║                                                                              ║
║  DEĞİŞİKLİK 4 — GERİ BESLEMELİ İTERATİF TRAŞLAMA                         ║
║    Önce temkinli (yüksek t_white), yeterli halo temizlenemediyse           ║
║    t_white'ı adım adım düşür, tekrar taşla.                                 ║
║    (V3/V4'te yanlış yöndeydi: V4 max_depth artırıyordu,                    ║
║     V3 t_white artırıyordu. V5: t_white düşürüyor = daha agresif)          ║
║                                                                              ║
║  KORUNANLAR (V4.1'den):                                                      ║
║    - FG_TO_MAX_DEPTH_TABLE (orijinal görüntülerden kalibre)                 ║
║    - 8 yönlü ışın + erozyon tabanlı sınır                                  ║
║    - Halo kalite istatistikleri (halo_mean, halo_p5...)                     ║
║    - multiprocessing ile tam CPU kullanımı                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import binary_erosion
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           AYARLAR                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

INPUT_DIR  = r"C:\Users\Yusuf Soylu\Desktop\Beyin Traslama\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\yedek\SiyahVeKaymaTemizBrainMaskUyg"
OUTPUT_DIR = r"C:\Users\Yusuf Soylu\Desktop\Beyin Traslama\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V5"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        TRAŞLAMA PARAMETRELERİ                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Dilim boyutu sınırları ────────────────────────────────────────────────────
VERY_THIN_MAX_FG_PCT = 2.0
THIN_MAX_FG_PCT      = 8.0

# ── Histogram bölge sınırı (DEĞİŞİKLİK 1) ────────────────────────────────────
# 180 altı = beyin dokusu bölgesi
# 180 üstü = kafatası kemiği bölgesi
# Literatür: beyin dokusu 20-80 HU (gerçek), normalize 8-bit'te ~51-179
# Kafatası kemiği 400-1000 HU (gerçek), normalize'de 180+
BRAIN_ZONE_MAX = 179   # 179 dahil = beyin bölgesi üst sınırı
SKULL_ZONE_MIN = 180   # 180 dahil = kafatası bölgesi alt sınırı

# ── Histogram düzleştirme ────────────────────────────────────────────────────
PEAK_SMOOTHING_WINDOW = 11   # Savitzky-Golay penceresi
PEAK_PROMINENCE       = 10   # Minimum pik belirginliği
PEAK_DISTANCE         = 10   # İki pik arası minimum mesafe (aynı bölge içi)

# ── Vadi kalite kontrolü ──────────────────────────────────────────────────────
# Vadi beyin tepesinin bu oranından yüksekse → düz histogram → fallback
VALLEY_MAX_RATIO = 0.60

# ── img_p95 tabanlı güvenlik sınırı (DEĞİŞİKLİK 3) ───────────────────────────
# t_white >= img_p95_fg + SAFETY_MARGIN garantisi
# Önplan piksellerinin %95'inin üstü + 20 HU = kafatası bölgesi başlangıcı
SAFETY_MARGIN = 20

# ── Fallback t_white ──────────────────────────────────────────────────────────
FALLBACK_FACTOR = 0.92
FALLBACK_MIN    = 220

# ── Dinamik MAX_DEPTH tablosu (V4.1'den kalibre, korunuyor) ──────────────────
FG_TO_MAX_DEPTH_TABLE = [
    ( 2.0,  40),
    ( 5.0,  83),
    ( 8.0, 104),
    (15.0, 130),
    (25.0, 160),
    (35.0, 193),
    (100.0,200),
]

# ── Güvenlik sınırları ────────────────────────────────────────────────────────
MIN_MAX_DEPTH  = 8
T_WHITE_MARGIN = 5      # t_white = min(t_white, img_max - T_WHITE_MARGIN)
N_DIRECTIONS   = 8

# ── İteratif feedback (DEĞİŞİKLİK 4) ─────────────────────────────────────────
# Adım 1: Temkinli başla (yüksek t_white = az sil)
# Adım 2: Yeterli halo temizlenemediyse t_white'ı düşür = daha agresif
# Adım 3: Tekrar taşla, kalite kontrol et
# t_white bu aralıkta aranır: [t_white_min, t_white_max]
ITERATIVE_STEP       = 5    # Her adımda t_white bu kadar düşer
ITERATIVE_MAX_ITER   = 6    # En fazla 6 iterasyon
# Hedef: fg'nin bu kadar silinmesi isteniyor
ITERATIVE_TARGET_PCT = 0.5  # % — minimum traşlama hedefi
# Kalite kapısı: silinen piksellerin ortalaması bu HU'dan düşükse dur
ITERATIVE_QUALITY_MIN_HU = 200  # halo_mean bu değerin altına düşerse dur


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      TEMEL YARDIMCI FONKSİYONLAR                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_t_bg(img: np.ndarray, border_width: int = 5) -> int:
    border = np.concatenate([
        img[:border_width, :].ravel(),
        img[-border_width:, :].ravel(),
        img[:, :border_width].ravel(),
        img[:, -border_width:].ravel(),
    ])
    return int(np.percentile(border, 99.9))


def classify_slice(fg_pct: float) -> str:
    if fg_pct <= VERY_THIN_MAX_FG_PCT:
        return "very_thin"
    elif fg_pct <= THIN_MAX_FG_PCT:
        return "thin"
    return "normal"


def get_dynamic_max_depth(fg_pct: float) -> int:
    for fg_upper, depth in FG_TO_MAX_DEPTH_TABLE:
        if fg_pct <= fg_upper:
            return max(depth, MIN_MAX_DEPTH)
    return 200


def _fallback_t_white(img_max: int) -> int:
    return max(int(FALLBACK_FACTOR * img_max), FALLBACK_MIN)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              DEĞİŞİKLİK 1 — KONUM BAZLI PİK TESPİTİ                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def detect_t_white_v5(img: np.ndarray, fg_mask: np.ndarray,
                       img_max: int) -> tuple:
    """
    Konum bazlı pik tespiti ile t_white hesapla.

    Yöntem:
      1. Önplan piksellerinin histogramını al, Savitzky-Golay ile düzelt
      2. Histogramı iki bölgeye ayır:
         brain_zone: [SKULL_ZONE_MIN altı] → beyin dokusu
         skull_zone: [SKULL_ZONE_MIN ve üstü] → kafatası kemiği
      3. Her bölgede en yüksek piki bul
      4. skull_zone'da pik yoksa → fallback
      5. İki pik arasındaki minimum = t_white
      6. Vadi kalite kontrolü

    V4.x'ten fark: "en büyük 2 pik" yerine "konuma göre brain/skull"
    Bu sayede beynin gri/beyaz madde alt bantları karışmıyor.

    Döndürür: (t_white, method, debug)
    """
    debug = {}
    fg_pixels = img[fg_mask].astype(np.int32)

    if len(fg_pixels) < 50:
        return _fallback_t_white(img_max), "fallback_too_few_pixels", debug

    # Histogram
    hist, _ = np.histogram(fg_pixels, bins=256, range=(0, 255))

    # Savitzky-Golay düzleştirme
    try:
        hist_s = savgol_filter(hist.astype(float),
                               window_length=PEAK_SMOOTHING_WINDOW,
                               polyorder=2)
        hist_s = np.clip(hist_s, 0, None)
    except Exception:
        hist_s = np.convolve(hist.astype(float), np.ones(7)/7, mode='same')

    # ── Bölge 1: Beyin zonu (0 - BRAIN_ZONE_MAX) ─────────────────────────────
    brain_zone = hist_s[:SKULL_ZONE_MIN]   # 0-179 arası

    # Beyin zonunda pik bul
    brain_peaks, _ = find_peaks(brain_zone,
                                 distance=PEAK_DISTANCE,
                                 prominence=PEAK_PROMINENCE)

    if len(brain_peaks) == 0:
        # Beyin piki yoksa → beyin zonu maksimumunu al (pik olmasa bile)
        brain_peak = int(np.argmax(brain_zone))
    else:
        # En yüksek beyin pikini seç
        brain_peak = int(brain_peaks[np.argmax(brain_zone[brain_peaks])])

    # ── Bölge 2: Kafatası zonu (SKULL_ZONE_MIN - 255) ─────────────────────────
    skull_zone = hist_s[SKULL_ZONE_MIN:]   # 180-255 arası

    # Kafatası zonunda pik bul
    skull_peaks_local, _ = find_peaks(skull_zone,
                                       distance=PEAK_DISTANCE,
                                       prominence=PEAK_PROMINENCE)

    debug['brain_peak'] = brain_peak
    debug['n_skull_peaks'] = len(skull_peaks_local)

    if len(skull_peaks_local) == 0:
        # Kafatası zonu tamamen düz veya pik yok
        # → Bu slice'ta gerçek kafatası pikseli çok az
        # → Fallback (V2 formülü)
        debug['fail'] = 'no_skull_peak_in_zone'
        return _fallback_t_white(img_max), "fallback_no_skull_peak", debug

    # En yüksek kafatası pikini seç (gerçek HU konumuna çevir: +SKULL_ZONE_MIN)
    skull_peak_local = int(skull_peaks_local[np.argmax(skull_zone[skull_peaks_local])])
    skull_peak = skull_peak_local + SKULL_ZONE_MIN

    debug['skull_peak'] = skull_peak
    debug['separation'] = skull_peak - brain_peak

    # ── Vadi: brain_peak ile skull_peak arasındaki minimum ───────────────────
    if skull_peak <= brain_peak:
        # Mantıksal tutarsızlık (olmamalı ama savunmacı)
        debug['fail'] = 'skull_below_brain'
        return _fallback_t_white(img_max), "fallback_skull_below_brain", debug

    segment    = hist_s[brain_peak: skull_peak + 1]
    valley_off = int(np.argmin(segment))
    valley_pos = brain_peak + valley_off

    brain_h    = float(hist_s[brain_peak])
    valley_h   = float(hist_s[valley_pos])
    valley_ratio = (valley_h / brain_h) if brain_h > 0 else 1.0

    debug['valley_pos']   = valley_pos
    debug['valley_ratio'] = round(valley_ratio, 3)

    # ── Vadi kalite kontrolü ──────────────────────────────────────────────────
    if valley_ratio > VALLEY_MAX_RATIO:
        # Vadi çok sığ → iki bölge arasında net ayrım yok
        # Bu genellikle kafatası çok az olan veya normalizasyon farklı slice
        debug['fail'] = f'shallow_valley ratio={valley_ratio:.2f}'
        return _fallback_t_white(img_max), "fallback_shallow_valley", debug

    # ── t_white = vadi konumu ─────────────────────────────────────────────────
    t_white = int(valley_pos)

    # Güvenlik kırpması 1: img_max sınırı
    t_white = min(t_white, img_max - T_WHITE_MARGIN)

    # Güvenlik kırpması 2: mutlak alt sınır (DEĞİŞİKLİK 3 — img_p95 tabanlı)
    # Bu slice'ta önplan piksellerinin p95'i nerede?
    fg_p95 = float(np.percentile(fg_pixels, 95))
    t_white_floor = int(fg_p95) + SAFETY_MARGIN
    debug['fg_p95']       = round(fg_p95, 1)
    debug['t_white_floor'] = t_white_floor

    if t_white < t_white_floor:
        # Vadi çok alçakta → beyin dokusu silinme riski
        # t_white_floor'a çek ama yöntemi "adjusted" olarak işaretle
        debug['t_white_raw'] = t_white
        t_white = t_white_floor
        return t_white, "two_peak_valley_adjusted", debug

    return t_white, "two_peak_valley", debug


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     RAY-MARCHING (V4.1'DEN KORUNUYOR)                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def find_fg_boundary(fg_mask: np.ndarray) -> list:
    eroded   = binary_erosion(fg_mask)
    boundary = fg_mask & ~eroded
    ys, xs   = np.where(boundary)
    return list(zip(ys.tolist(), xs.tolist()))


def ray_march(img: np.ndarray, start_y: int, start_x: int,
              dy: float, dx: float, t_white: int,
              max_depth: int, fg_mask: np.ndarray) -> np.ndarray:
    h, w   = img.shape
    mask   = np.zeros((h, w), dtype=bool)
    depth  = 0
    cy, cx = float(start_y), float(start_x)

    while True:
        cy += dy; cx += dx; depth += 1
        iy, ix = int(round(cy)), int(round(cx))

        if iy < 0 or iy >= h or ix < 0 or ix >= w: break
        if not fg_mask[iy, ix]:                      break
        if depth > max_depth:                         break

        if img[iy, ix] >= t_white:
            mask[iy, ix] = True
        else:
            break

    return mask


def shave_slice(img: np.ndarray, fg_mask: np.ndarray,
                t_white: int, max_depth: int) -> np.ndarray:
    h, w = img.shape
    final = np.zeros((h, w), dtype=bool)

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
                continue
            final |= ray_march(img, sy, sx, dy, dx,
                                t_white, max_depth, fg_mask)
    return final


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              DEĞİŞİKLİK 4 — İTERATİF GERİ BESLEMELİ TRAŞLAMA             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def iterative_shave(img: np.ndarray, fg_mask: np.ndarray,
                     t_white_init: int, max_depth: int,
                     fg_px: int) -> tuple:
    """
    İteratif traşlama: temkinli başla, yetmezse daha agresif ol.

    Strateji:
      - t_white_init'ten başla (histogram vadisinden veya fallback)
      - Eğer halo_mean >= ITERATIVE_QUALITY_MIN_HU VE shave_pct >= ITERATIVE_TARGET_PCT
        → kabul et
      - Yeterli silme olmadıysa t_white'ı ITERATIVE_STEP kadar düşür
      - halo_mean < ITERATIVE_QUALITY_MIN_HU → beyin siliniyor, dur
      - En fazla ITERATIVE_MAX_ITER iterasyon

    V4'ten fark:
      V4: az sildiyse max_depth artır (derine git)
      V5: az sildiyse t_white düşür (eşiği agresifleştir)
      İkisi birlikte de çalışabilir ama önce t_white ayarı daha temkinli.

    Döndürür: (shave_mask, final_t_white, n_iterations, stop_reason)
    """
    t_white    = t_white_init
    best_mask  = None
    best_tw    = t_white_init
    stop_reason = "max_iter"

    for iteration in range(ITERATIVE_MAX_ITER + 1):

        shave_mask = shave_slice(img, fg_mask, t_white, max_depth)
        shaved_px  = int(shave_mask.sum())
        shave_pct  = shaved_px / max(fg_px, 1) * 100.0

        # Halo kalite kontrolü: silinen piksellerin yoğunluk ortalaması
        if shaved_px > 0:
            halo_pixels = img[shave_mask].astype(np.float32)
            halo_mean_cur = float(np.mean(halo_pixels))
        else:
            halo_mean_cur = 255.0   # hiç silinmediyse sorun değil

        # Kalite kapısı: beyin siliniyorsa dur
        if shaved_px > 0 and halo_mean_cur < ITERATIVE_QUALITY_MIN_HU:
            stop_reason = "quality_gate"
            # Bir önceki iterasyonun maskesini kullan (daha iyi)
            if best_mask is not None:
                shave_mask = best_mask
                t_white    = best_tw
            break

        # Hedef sağlandıysa kabul et
        if shave_pct >= ITERATIVE_TARGET_PCT:
            best_mask  = shave_mask
            best_tw    = t_white
            stop_reason = "target_reached"
            break

        # Limit doldu
        if iteration == ITERATIVE_MAX_ITER:
            if best_mask is not None:
                shave_mask = best_mask
                t_white    = best_tw
            stop_reason = "max_iter"
            break

        # En iyi sonucu sakla
        if best_mask is None or shave_pct > (best_mask.sum() / max(fg_px,1) * 100):
            best_mask = shave_mask
            best_tw   = t_white

        # t_white'ı düşür → daha agresif
        t_white = max(t_white - ITERATIVE_STEP,
                      int(np.percentile(img[fg_mask], 95)) + SAFETY_MARGIN,
                      120)

        # t_white azaltılamıyorsa dur
        if t_white >= t_white_init - ITERATIVE_STEP * (iteration + 1) + ITERATIVE_STEP:
            stop_reason = "t_white_floor"
            break

    return shave_mask, t_white, iteration, stop_reason


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     İSTATİSTİK FONKSİYONLARI                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def px_stats(pixels: np.ndarray, prefix: str) -> dict:
    if len(pixels) == 0:
        keys = ['count','mean','std','min','p5','p10','p25','p50','p75','p90','p95','max']
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       ANA İŞLEM FONKSİYONU                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def process_single_image(img_path: Path, halo_dir: Path) -> tuple:
    log = {'filename': img_path.name}

    try:
        pil_img  = Image.open(img_path).convert("L")
        img      = np.array(pil_img, dtype=np.uint8)
        h, w     = img.shape
        total_px = h * w
        img_max  = int(img.max())

        log.update({'img_max': img_max, 'total_px': total_px})

        # ── Ön plan maskesi ──────────────────────────────────────────────────
        t_bg    = get_t_bg(img)
        fg_mask = img > t_bg
        fg_px   = int(fg_mask.sum())
        fg_pct  = fg_px / total_px * 100.0

        slice_class = classify_slice(fg_pct)
        max_depth   = get_dynamic_max_depth(fg_pct)

        log.update({'t_bg': t_bg, 'fg_px': fg_px,
                    'fg_pct': round(fg_pct, 4),
                    'slice_class': slice_class,
                    'max_depth': max_depth})

        if fg_px == 0:
            log['skip_reason'] = 'no_foreground'
            halo_arr = np.zeros((h, w), dtype=np.uint8)
            Image.fromarray(halo_arr).save(str(halo_dir / img_path.name))
            return img.copy(), halo_arr, log

        # ── t_white tespiti (DEĞİŞİKLİK 1) ──────────────────────────────────
        t_white, t_method, t_debug = detect_t_white_v5(img, fg_mask, img_max)

        # Güvenlik kırpması
        t_white = min(t_white, img_max - T_WHITE_MARGIN)

        log.update({
            't_white_init' : t_white,
            't_white_method': t_method,
            **{f'peak_{k}': v for k, v in t_debug.items()}
        })

        # ── İteratif traşlama (DEĞİŞİKLİK 4) ────────────────────────────────
        shave_mask, t_white_final, n_iters, stop_reason = iterative_shave(
            img=img, fg_mask=fg_mask,
            t_white_init=t_white,
            max_depth=max_depth,
            fg_px=fg_px,
        )

        shaved_px  = int(shave_mask.sum())
        shaved_pct = shaved_px / max(fg_px, 1) * 100.0

        log.update({
            't_white_final'     : t_white_final,
            'n_iterations'      : n_iters,
            'stop_reason'       : stop_reason,
            'shaved_px'         : shaved_px,
            'shaved_pct_of_fg'  : round(shaved_pct, 3),
        })

        # ── Traşlanmış görüntü ───────────────────────────────────────────────
        output_img = img.copy()
        output_img[shave_mask] = 0

        # ── Halo mask ────────────────────────────────────────────────────────
        halo_mask_bool = (img > 0) & (output_img == 0)
        halo_uint8     = halo_mask_bool.astype(np.uint8) * 255

        halo_px  = int(halo_mask_bool.sum())
        halo_pct = halo_px / total_px * 100.0

        log.update({'halo_px': halo_px, 'halo_pct': round(halo_pct, 4)})

        Image.fromarray(halo_uint8).save(str(halo_dir / img_path.name))

        # ── Halo istatistikleri ──────────────────────────────────────────────
        halo_pixels = img[halo_mask_bool].astype(np.uint8)
        log.update(px_stats(halo_pixels, 'halo'))

        return output_img, halo_uint8, log

    except Exception as e:
        log['error'] = str(e)
        try:
            img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        except Exception:
            img = np.zeros((512, 512), dtype=np.uint8)
        halo_empty = np.zeros_like(img)
        return img, halo_empty, log


def _worker(args):
    """multiprocessing worker — (img_path, halo_dir) → log dict"""
    img_path, halo_dir = args
    output_img, halo_arr, log = process_single_image(img_path, halo_dir)
    # Görüntüyü kaydet (worker içinde)
    if 'error' not in log:
        out_path = halo_dir.parent / img_path.name
        Image.fromarray(output_img).save(str(out_path))
    return log


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              ANA DÖNGÜ                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_excel(all_rows: list, output_excel: Path) -> None:
    df = pd.DataFrame(all_rows).sort_values('filename').reset_index(drop=True)
    df['patient_id'] = df['filename'].str.extract(r'^(\d+)_').astype(str)

    with pd.ExcelWriter(str(output_excel), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Tüm Dilimler', index=False)

        agg = {k: v for k, v in {
            'filename': 'count', 'fg_pct': 'mean',
            'shaved_pct_of_fg': 'mean', 'max_depth': 'mean',
            'halo_px': 'sum', 'halo_mean': 'mean',
        }.items() if k in df.columns}
        hasta = df.groupby('patient_id', as_index=False).agg(agg)
        hasta.rename(columns={'filename': 'n_slices'}, inplace=True)
        hasta.to_excel(writer, sheet_name='Hasta Özeti', index=False)

        if 'slice_class' in df.columns:
            agg2 = {k: v for k, v in {
                'filename': 'count', 'fg_pct': 'mean',
                'shaved_pct_of_fg': 'mean', 'halo_px': 'mean',
                'halo_mean': 'mean',
            }.items() if k in df.columns}
            sinif = df.groupby('slice_class', as_index=False).agg(agg2)
            sinif.rename(columns={'filename': 'n_slices'}, inplace=True)
            sinif.to_excel(writer, sheet_name='Dilim Sınıfı Özeti', index=False)

    print(f"  Excel: {output_excel.name}  ({len(df)} satır)")


def main():
    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    halo_path   = output_path / "halo_mask_png"
    excel_path  = output_path / "v5_istatistik.xlsx"

    output_path.mkdir(parents=True, exist_ok=True)
    halo_path.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_path.glob("*.png"))
    if not png_files:
        print(f"HATA: {INPUT_DIR} içinde PNG bulunamadı!")
        return

    n_cpu = mp.cpu_count()
    n_workers = max(1, n_cpu - 1)

    print(f"\n{'='*72}")
    print(f"  TRAŞLAMA V5  |  {len(png_files)} dosya  |  {n_workers}/{n_cpu} CPU")
    print(f"{'='*72}")
    print(f"  [DEĞ.1] Konum bazlı pik tespiti: brain_zone=0-{BRAIN_ZONE_MAX}, skull_zone={SKULL_ZONE_MIN}-255")
    print(f"  [DEĞ.3] img_p95 güvenlik sınırı: t_white >= fg_p95 + {SAFETY_MARGIN}")
    print(f"  [DEĞ.4] İteratif feedback: t_white adım={ITERATIVE_STEP}, max_iter={ITERATIVE_MAX_ITER}")
    print(f"  Kalite kapısı: halo_mean >= {ITERATIVE_QUALITY_MIN_HU} HU\n")

    args_list = [(p, halo_path) for p in png_files]

    all_logs = []
    done = 0
    total = len(png_files)

    with mp.Pool(processes=n_workers) as pool:
        for log in pool.imap_unordered(_worker, args_list, chunksize=4):
            all_logs.append(log)
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  ⏳ {done}/{total} ({done/total*100:.1f}%) tamamlandı...")

    # Özet
    valid = [r for r in all_logs if 'error' not in r]
    errors = [r for r in all_logs if 'error' in r]

    methods = {}
    for r in valid:
        m = r.get('t_white_method', 'unknown')
        methods[m] = methods.get(m, 0) + 1

    halo_means = [r.get('halo_mean', np.nan) for r in valid if r.get('halo_px',0)>0]
    halo_means = [x for x in halo_means if not np.isnan(x)]

    print(f"\n{'='*72}")
    print(f"  TAMAMLANDI")
    print(f"  Başarılı : {len(valid)}  |  Hata: {len(errors)}")
    print(f"\n  t_white yöntemleri:")
    for m, cnt in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"    {m:35s}: {cnt:4d} (%{cnt/len(valid)*100:.1f})")
    print(f"\n  Halo kalitesi (halo_px>0 olanlar):")
    print(f"    halo_mean ort : {np.mean(halo_means):.1f} HU  (hedef: ≥220)")
    print(f"    halo_mean < 200 (risk): {sum(1 for x in halo_means if x < 200)}")
    print(f"\n  Ort. halo_px : {np.mean([r.get('halo_px',0) for r in valid]):.1f}")
    print(f"  Hiç silinmeyen: {sum(1 for r in valid if r.get('halo_px',0)==0)}")

    build_excel(all_logs, excel_path)
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()