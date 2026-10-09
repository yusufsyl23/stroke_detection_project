# -*- coding: utf-8 -*-
"""
BRAIN MASK GENERATION SCRIPT
==============================
Dataset analizine gore belirlenmis parametreler:
  - 35 hasta, 1253 slice
  - Normal resimler: Otsu 50-65 araliginda yogunlasiyor
  - Problem grup: Otsu = 0.46875 (93 resim) -> mean-based fallback
  - Bos sliceler: 155 resim -> siyah maske
  - Hibrit threshold: MIN=45, MAX=70

TURKCE KARAKTER DESTEGI:
  - Path islemleri pathlib.Path ile yapiliyor (unicode native)
  - Resim okuma: open(path, 'rb') + PIL.Image.open(BytesIO(...))
    -> Windows'ta Turkce klasor/dosya adlari sorunsuz calisir
  - Resim kaydetme: gecici dosyaya yazip shutil ile tasiyor
    -> alternatif olarak numpy -> PIL -> BytesIO -> dosya
"""

import os
import sys
import io
import time
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Windows stdout encoding (cmd/PowerShell Turkce karakter icin)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python 3.6 compatibility

# PIL - binary mode okuma ile Turkce yol destegi
try:
    from PIL import Image
except ImportError:
    print("HATA: Pillow bulunamadi. Cozum: pip install Pillow")
    sys.exit(1)

try:
    from skimage.filters import threshold_otsu
except ImportError:
    print("HATA: scikit-image bulunamadi. Cozum: pip install scikit-image")
    sys.exit(1)

try:
    from scipy.ndimage import label as scipy_label
except ImportError:
    print("HATA: scipy bulunamadi. Cozum: pip install scipy")
    sys.exit(1)


# ============================================================
# PARAMETRELER — Dataset analizine gore belirlenmis
# ============================================================

# Klasor yollarini buraya gir (Turkce karakter sorun cikarmiyor)
INPUT_DIR  = "C:/Users/Yusuf Soylu/Desktop/GPT Sil Baştan/Eren Sİyah ve Kaymadan Arındırılmış Veri Seti/SiyahVeKaymaTemizBrain_Algoritma_Cikti"
OUTPUT_DIR = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Algoritma_Brain_Mask_Gürültülü"

# Hibrit threshold sinirlari (normal grup P10=48, P90=65)
MIN_THRESHOLD = 45
MAX_THRESHOLD = 70

# Problem grup esigi: Otsu bu degerden kucukse mean-based fallback devreye girer
OTSU_FALLBACK_LIMIT = 20

# Mean-based fallback katsayisi: threshold = image_mean * bu deger
MEAN_MULTIPLIER = 0.40

# Bos slice tespiti (YENİ - DAHA GUVENİLİR YONTEMi):
#   Eski: dark_ratio > 0.95 → HATALI (slice0009 gibi az beyin iceren resimleri siliyor)
#   Yeni: max_pixel < 5    → kesin bos (hizli, gercek boslari yakalar)
#         + threshold sonrasi beyin_alan < %0.3 → bos say (ikinci guvenlik kati)
# Neden daha iyi?
#   Bimodal resimler: %97 siyah AMA kalan %3 gercek beyin → dark_ratio yanlis sayiyordu
#   max_pixel kontrolu: eger resimde hic parlak piksel yoksa (max<5) kesinlikle bos
EMPTY_MAX_PIXEL = 5       # max piksel bu degerden kucukse kesin bos
EMPTY_BRAIN_RATIO = 0.003 # threshold sonrasi beyaz alan bu yuzdenin altindaysa bos


# Paralel islem sayisi (None = sistemdeki CPU sayisi)
NUM_WORKERS = None


# ============================================================
# YARDIMCI FONKSIYONLAR
# ============================================================

def load_image_safe(path: Path) -> np.ndarray:
    """
    Turkce/ozel karakterli yollardan guvenli resim yukleme.
    Yontem: path.read_bytes() ile ham byte okuma
            -> BytesIO ile PIL'e ver -> numpy array
    Bu sekilde Windows'un dosya acma limitlerini atliyoruz.
    """
    raw_bytes = path.read_bytes()           # Python pathlib -> unicode native
    img = Image.open(io.BytesIO(raw_bytes)).convert('L')  # grayscale
    return np.array(img, dtype=np.float32)


def save_image_safe(array: np.ndarray, path: Path):
    """
    Turkce/ozel karakterli yollara guvenli kaydetme.
    Yontem: PIL Image -> BytesIO buffer -> path.write_bytes()
    """
    img = Image.fromarray(array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    path.write_bytes(buffer.getvalue())


# ============================================================
# THRESHOLD HESAPLAMA
# ============================================================

def compute_hybrid_threshold(image_array: np.ndarray):
    """
    Hibrit threshold stratejisi:

    1. Bos slice?     -> None, "EMPTY"
    2. Otsu < 20?     -> mean-based fallback (bimodal bozulma)
    3. Normal?        -> Otsu, [MIN, MAX] araligina kisitla

    Returns:
        (threshold: float | None, method: str)
    """
    flat = image_array.flatten()

    # --- Bos slice tespiti: max_pixel kontrolu ---
    # dark_ratio yerine max_pixel kullaniyoruz
    # Sebep: bazi sliceler bimodal (%97 siyah + %3 gercek beyin)
    # dark_ratio bunlari yanlis "bos" sayiyordu → maskeler siyah cikiyordu
    if float(np.max(flat)) < EMPTY_MAX_PIXEL:
        return None, "EMPTY"

    # --- Otsu hesapla ---
    try:
        otsu_val = float(threshold_otsu(image_array))
    except Exception:
        otsu_val = 0.0

    # --- Problem grup: bimodal dagilim bozulmasi ---
    # Bu resimlerde piksel dagiliminda 30-80 araliginda hic piksel yok
    # Otsu gecis bolgesi bulamiyor -> 0.46875 gibi yanlis deger uretir
    if otsu_val < OTSU_FALLBACK_LIMIT:
        mean_val = float(np.mean(flat))
        fallback_t = mean_val * MEAN_MULTIPLIER
        threshold = max(fallback_t, float(MIN_THRESHOLD))
        method = f"FALLBACK(mean={mean_val:.1f}->raw={fallback_t:.1f}->final={threshold:.1f})"

    # --- Normal grup: Otsu var, [MIN, MAX] sinirla ---
    else:
        threshold = float(np.clip(otsu_val, MIN_THRESHOLD, MAX_THRESHOLD))
        if otsu_val < MIN_THRESHOLD:
            method = f"CLAMPED_LOW(otsu={otsu_val:.1f}->{threshold:.1f})"
        elif otsu_val > MAX_THRESHOLD:
            method = f"CLAMPED_HIGH(otsu={otsu_val:.1f}->{threshold:.1f})"
        else:
            method = f"OTSU({otsu_val:.1f})"

    return threshold, method


# ============================================================
# MASKE URETME ADIMLARI
# ============================================================

def flood_fill_holes(binary: np.ndarray) -> np.ndarray:
    """
    Beyin maskesindeki ic delikleri (ventrikul, sulkus, damar) kapat.

    Neden flood fill, neden morfoloji degil?
    - Morfoloji (dilate/erode): sinir sekli bozulur, kernel boyutu kritik
    - Flood fill: sadece baglantisalligi kullanir, sinirlara dokunmaz

    Adimlar:
      1. Resmi 1 piksel genislet (kose her zaman dis arka plan)
      2. Kose pikseline temas eden bagli bilesken = dis arka plan
      3. binary=0 VE dis arka planda DEGIL = ic delik
      4. Ic delikleri beyaz yap
    """
    padded = np.pad(binary.astype(np.uint8), 1, constant_values=0)
    background = (padded == 0)
    labeled, _ = scipy_label(background)
    outer_label = labeled[0, 0]          # kose = dis arka plan etiketi
    outer_mask = (labeled == outer_label)[1:-1, 1:-1]

    filled = binary.copy()
    internal_holes = (~binary) & (~outer_mask)
    filled[internal_holes] = True
    return filled





# ============================================================
# TEK RESIM PIPELINE
# ============================================================

def process_single_image(args):
    """
    Tek resim icin tam islem akisi.
    ProcessPoolExecutor'a gonderilecegi icin module-level fonksiyon.

    Returns:
        (filename, status, method, brain_pct, error_msg)
    """
    input_path_str, output_path_str = args
    input_path  = Path(input_path_str)
    output_path = Path(output_path_str)
    filename    = input_path.name

    try:
        # 1) Yukle (Turkce yol destekli)
        arr = load_image_safe(input_path)

        # 2) Threshold hesapla
        threshold, method = compute_hybrid_threshold(arr)

        # 3) Bos slice -> siyah maske
        if threshold is None:
            empty_mask = np.zeros(arr.shape, dtype=np.uint8)
            save_image_safe(empty_mask, output_path)
            return (filename, "EMPTY", "EMPTY", 0.0, None)

        # 4) Binary threshold
        binary = arr > threshold

        # 5) Flood fill: ic delikleri kapat
        binary = flood_fill_holes(binary)


        # 6) Threshold sonrasi bos kontrol (ikinci guvenlik kati)
        #    Threshold uygulandiktan sonra beyin cok azsa gercekten bos demek
        brain_ratio_check = float(binary.mean())
        if brain_ratio_check < EMPTY_BRAIN_RATIO:
            empty_mask = np.zeros(arr.shape, dtype=np.uint8)
            save_image_safe(empty_mask, output_path)
            return (filename, "EMPTY", method + "[post-check]", 0.0, None)

        # 7) Kaydet (Turkce yol destekli)
        mask_uint8 = binary.astype(np.uint8) * 255
        save_image_safe(mask_uint8, output_path)

        brain_pct = float(binary.mean() * 100)
        return (filename, "OK", method, brain_pct, None)

    except Exception as e:
        return (filename, "ERROR", "—", 0.0, str(e))


# ============================================================
# ANA FONKSIYON
# ============================================================

def main():
    input_path  = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    # --- Girdi kontrolu ---
    if not input_path.exists():
        print(f"HATA: Input klasoru bulunamadi:")
        print(f"  {INPUT_DIR}")
        print(f"\nIpucu: Klasor yolundaki Turkce karakterleri kontrol edin.")
        print(f"Scriptini 'INPUT_DIR' satirini duzenlemeniz gerekebilir.")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # --- Dosya listesi ---
    png_files = sorted(input_path.glob("*.png"))
    total = len(png_files)

    if total == 0:
        print(f"HATA: Klasorde .png dosyasi bulunamadi:")
        print(f"  {INPUT_DIR}")
        sys.exit(1)

    # --- Baslik ---
    print(f"\n{'='*62}")
    print(f"  BRAIN MASK GENERATION")
    print(f"{'='*62}")
    print(f"  Toplam resim      : {total}")
    print(f"  Input             : {INPUT_DIR}")
    print(f"  Output            : {OUTPUT_DIR}")
    print(f"{'─'*62}")
    print(f"  Min threshold     : {MIN_THRESHOLD}")
    print(f"  Max threshold     : {MAX_THRESHOLD}")
    print(f"  Fallback tetigi   : Otsu < {OTSU_FALLBACK_LIMIT}  ->  mean x {MEAN_MULTIPLIER}")
    print(f"  Bos slice esigi   : max_pixel < {EMPTY_MAX_PIXEL}  |  post-threshold beyin < {EMPTY_BRAIN_RATIO*100:.1f}%")
    print(f"{'='*62}\n")

    # --- Gorev listesi (str olarak gecmek gerekiyor - pickle icin) ---
    tasks = [
        (str(p), str(output_path / p.name))
        for p in png_files
    ]

    # --- Paralel isleme ---
    start_time = time.time()
    results    = []
    counters   = {
        "OK": 0, "EMPTY": 0, "ERROR": 0,
        "OTSU": 0, "FALLBACK": 0, "CLAMPED_LOW": 0, "CLAMPED_HIGH": 0
    }

    workers = NUM_WORKERS or os.cpu_count()
    print(f"  Islem basliyor ({workers} core)...\n")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures   = {executor.submit(process_single_image, t): t for t in tasks}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            filename, status, method, brain_pct, err = future.result()
            results.append((filename, status, method, brain_pct, err))

            # Sayaclari guncelle
            counters[status] = counters.get(status, 0) + 1
            if status == "OK":
                if "FALLBACK"     in method: counters["FALLBACK"]     += 1
                elif "CLAMPED_LOW"  in method: counters["CLAMPED_LOW"]  += 1
                elif "CLAMPED_HIGH" in method: counters["CLAMPED_HIGH"] += 1
                else:                          counters["OTSU"]          += 1

            # Ilerleme cubugu
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                eta     = (elapsed / completed) * (total - completed) if completed < total else 0
                pct     = completed / total * 100
                filled  = int(pct / 4)
                bar     = '#' * filled + '-' * (25 - filled)
                print(f"  [{bar}] {pct:5.1f}%  {completed:4d}/{total}  ETA:{eta:4.0f}s", end='\r')

    elapsed = time.time() - start_time
    print(f"\n\n{'='*62}")
    print(f"  TAMAMLANDI  —  {elapsed:.1f} saniye  ({elapsed/60:.1f} dakika)")
    print(f"{'='*62}")
    print(f"  Maske uretildi   : {counters['OK']}")
    print(f"  Bos slice (siyah): {counters['EMPTY']}")
    print(f"  Hata             : {counters['ERROR']}")
    print(f"{'─'*62}")
    print(f"  Threshold yontem dagilimlari (basarili):")
    print(f"  -> Normal Otsu          : {counters['OTSU']}")
    print(f"  -> Mean fallback (<{OTSU_FALLBACK_LIMIT}) : {counters['FALLBACK']}")
    print(f"  -> Clamped low  (<{MIN_THRESHOLD}) : {counters['CLAMPED_LOW']}")
    print(f"  -> Clamped high (>{MAX_THRESHOLD}) : {counters['CLAMPED_HIGH']}")

    # Beyin alani istatistigi
    brain_pcts = [r[3] for r in results if r[1] == "OK"]
    if brain_pcts:
        bp = np.array(brain_pcts)
        print(f"{'─'*62}")
        print(f"  Uretilen maskelerde beyin alani (%):")
        print(f"  -> Ortalama : {bp.mean():.1f}%")
        print(f"  -> Medyan   : {np.median(bp):.1f}%")
        print(f"  -> Min/Max  : {bp.min():.1f}% / {bp.max():.1f}%")

    # Hatali dosyalar
    errors = [(r[0], r[4]) for r in results if r[1] == "ERROR"]
    if errors:
        print(f"{'─'*62}")
        print(f"  HATALI DOSYALAR ({len(errors)} adet):")
        for fname, err in errors[:20]:
            print(f"    {fname}: {err}")
        if len(errors) > 20:
            print(f"    ... {len(errors)-20} tane daha (log dosyasina bakin)")

    # --- Log dosyasi ---
    log_path = output_path / "generation_log.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Brain Mask Generation Log\n")
        f.write(f"Tarih     : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Toplam    : {total}  |  OK: {counters['OK']}  |  EMPTY: {counters['EMPTY']}  |  ERROR: {counters['ERROR']}\n")
        f.write(f"Parametreler: MIN={MIN_THRESHOLD}  MAX={MAX_THRESHOLD}  FALLBACK_LIMIT={OTSU_FALLBACK_LIMIT}  MEAN_MULT={MEAN_MULTIPLIER}\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'DOSYA ADI':<35} {'DURUM':<8} {'BEYIN%':>7}  YONTEM\n")
        f.write("-" * 80 + "\n")
        for fname, status, method, bp, err in sorted(results):
            err_str = f"  HATA: {err}" if err else ""
            f.write(f"{fname:<35} {status:<8} {bp:>6.2f}%  {method}{err_str}\n")

    print(f"{'='*62}")
    print(f"  Log  : {log_path}")
    print(f"  Cikti: {OUTPUT_DIR}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()