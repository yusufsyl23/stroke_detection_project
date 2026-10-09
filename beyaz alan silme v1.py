import os
import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image


# ================================
# >>> KLASÖR YOLLARINI BURADA DEĞİŞTİR <
# ================================
INPUT_DIR  = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\SiyahVeKaymaTemizBrainMaskUyg")
OUTPUT_DIR = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V1\Önce Tekrarsız Sonra Traşlama V1")


# ================================
# Otomatik eşik hesaplama
# ================================
def auto_thresholds(img_u8: np.ndarray):
    border = np.concatenate([
        img_u8[:5, :].ravel(),
        img_u8[-5:, :].ravel(),
        img_u8[:, :5].ravel(),
        img_u8[:, -5:].ravel()
    ])
    t_bg    = int(np.percentile(border, 99.9))

    """
    Neden kenara bakıyor? BT görüntülerinde beyin her zaman ortada. Görüntünün 4 kenarı (5 piksel kalınlığında) neredeyse tamamen siyah 
    (0 değer) — orada sadece hava var. Ama bazen gürültü var: 1, 2, 3 gibi küçük değerler. percentile(99.9) diyor ki: 
    "Bu sınır piksellerin %99.9'unun altında olan değer nedir?" Çıkan sayı genellikle 0, bazen 1–5 arası çıkar.
    Sonuç nedir? t_bg = arka planın üst sınırı. Bu değer ve altı = hava/boşluk, üstü = gerçek doku. Algoritma ray-marching sırasında 
    bu sınırı "foreground'un başlangıcı" olarak kullanacak.
    """

    mx      = int(img_u8.max())
    t_white = int(max(200, 0.90 * mx))

    """
    Bu ALG1'in en kritik parametresi. "Hangi piksel değerinden itibaren 'kafatası kemiği' sayalım?" sorusunun cevabı.
    Mantık: Görüntünün en parlak pikseli `mx`. Onun %90'ı t_white'ı verir. Ama en az 200 olsun (güvenlik tabanı).
    Örnekle: Görüntünün maksimumu 255 ise → 0.90 × 255 = 229.5 → t_white = 229. Yani 229 ve üzeri değer = kafatası kemiği, sil.
    Bu sabit bir kuraldır — görüntüye göre değişmez. Her dilime, ister beyin büyük olsun ister küçük, 
    ister kafatası kalın olsun ister ince, hep aynı formül uygulanır. İşte ALG1'in temel kısıtlaması buradan gelir.
    """
    return t_bg, t_white


# ================================
# Ray-marching halo temizleme
# ================================

"""
Bu adım ALG1'in kalbi. "Ray-marching" demek: her satır ve her sütun boyunca kenardan içe doğru ışın gönderiyorsun, 
ışın kafatası pikselini gördüğü sürece ilerliyor ve o pikseli siliyor, beyin pikseline çarptığında duruyor.
Ama ALG1'in kritik bir farkı var: derinlik sınırı yok. Işın kafatası pikseli gördüğü sürece SONSUZA kadar ilerliyor.
ALG1'de tek bir `out` dizisi var — 4 yön aynı diziyi paylaşıyor. Bunu birazdan göreceğiz.
"""
def strip_halo(img_u8: np.ndarray):
    t_bg, t_white = auto_thresholds(img_u8)
    h, w = img_u8.shape
    out        = img_u8.copy()
    halo_mask  = np.zeros_like(out, dtype=np.uint8)

    """
    Her satır için (y = 0'dan 511'e):

        1) x=0'dan başla
        2) img[y,x] <= t_bg iken ilerle — arka planı atla
        3) img[y,x] >= t_white VE devam edebiliyorken:
             out[y,x] = SİL
             x += 1   (bir adım sağa)
        4) Koşul bozulunca dur
        
    Yani: "Kenardan gir, siyah olanları atla, beyaz olanları (kafatası) sil, gri (beyin) görünce dur."
    Derinlik sınırı YOK. Eğer o satırda beyin yoksa veya tüm satır beyazsa, algoritma görüntünün tam karşısına geçer — 512 piksel boyunca gider. 
    Bu durumda beyin pikselleri de silinir.
    Sağ → Sol, Yukarı → Aşağı, Aşağı → Yukarı
    Aynı mantık, 3 farklı yönden tekrar uygulanır. Toplamda beynin 4 kenarı da taranır.
    
    ALG1'in ikinci kritik sorunu: 4 yön aynı out dizisini paylaşıyor. Sol→Sağ bir pikseli silince, o piksel artık 0. 
    Sağ→Sol aynı konuma geldiğinde değer 0 görür — t_bg'den küçük — arka plan zanneder ve atlar. Yani daha önceden silinmiş pikseli 
    görüp "oh arka plan, foreground buradan başlıyor" diye yanılır ve yanlış noktadan başlar. Bu çift sayım / atlama hatasına yol açar.
    """

    # LEFT -> RIGHT
    for y in range(h):
        x = 0
        while x < w and out[y, x] <= t_bg:   x += 1
        while x < w and out[y, x] >= t_white: halo_mask[y, x] = 255; out[y, x] = 0; x += 1

    # RIGHT -> LEFT
    for y in range(h):
        x = w - 1
        while x >= 0 and out[y, x] <= t_bg:   x -= 1
        while x >= 0 and out[y, x] >= t_white: halo_mask[y, x] = 255; out[y, x] = 0; x -= 1

    # TOP -> BOTTOM
    for x in range(w):
        y = 0
        while y < h and out[y, x] <= t_bg:   y += 1
        while y < h and out[y, x] >= t_white: halo_mask[y, x] = 255; out[y, x] = 0; y += 1

    # BOTTOM -> TOP
    for x in range(w):
        y = h - 1
        while y >= 0 and out[y, x] <= t_bg:   y -= 1
        while y >= 0 and out[y, x] >= t_white: halo_mask[y, x] = 255; out[y, x] = 0; y -= 1

    return out, halo_mask, t_bg, t_white


# ================================
# Tek dosya işleme (worker)
# ================================
def process_file(file: Path, clean_dir: Path, mask_dir: Path):
    try:
        # PIL ile oku (Türkçe karakter / boşluk sorunsuz)
        pil_img = Image.open(file).convert("L")
        img     = np.array(pil_img, dtype=np.uint8)

        cleaned, halo_mask, t_bg, t_white = strip_halo(img)

        # PIL ile kaydet
        Image.fromarray(cleaned).save(clean_dir / file.name)
        Image.fromarray(halo_mask).save(mask_dir  / file.name)

        return {
            "filename"           : file.name,
            "height"             : img.shape[0],
            "width"              : img.shape[1],
            "min"                : int(img.min()),
            "max"                : int(img.max()),
            "mean"               : float(img.mean()),
            "std"                : float(img.std()),
            "p95"                : float(np.percentile(img, 95)),
            "p99"                : float(np.percentile(img, 99)),
            "t_bg"               : t_bg,
            "t_white"            : t_white,
            "halo_pixels_removed": int((halo_mask > 0).sum()),
            "error"              : "",
        }

    except Exception as e:
        return {
            "filename": file.name, "error": str(e),
            "height": 0, "width": 0, "min": 0, "max": 0,
            "mean": 0, "std": 0, "p95": 0, "p99": 0,
            "t_bg": 0, "t_white": 0, "halo_pixels_removed": 0,
        }


# ================================
# Ana program
# ================================
if __name__ == "__main__":
    clean_dir = OUTPUT_DIR / "cleaned_png"
    mask_dir  = OUTPUT_DIR / "halo_mask_png"
    clean_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(INPUT_DIR.glob("*.png"))
    print(f"Bulunan PNG sayısı: {len(png_files)}")

    # CPU çekirdek sayısı (1 çekirdek OS için bırak)
    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Kullanılan CPU çekirdeği: {n_workers} / {mp.cpu_count()}")

    worker = partial(process_file, clean_dir=clean_dir, mask_dir=mask_dir)

    rows = []
    done = 0
    total = len(png_files)

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(worker, png_files, chunksize=4):
            rows.append(result)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total} tamamlandı...")

    # Excel kaydet
    df         = pd.DataFrame(rows).sort_values("filename")
    excel_path = OUTPUT_DIR / "png_slice_statistics.xlsx"
    df.to_excel(excel_path, index=False)

    errors = df[df["error"] != ""]
    print(f"\nİşlem tamamlandı. Hata sayısı: {len(errors)}")
    if not errors.empty:
        print("Hatalı dosyalar:")
        print(errors[["filename", "error"]].to_string(index=False))

    print(f"Excel  : {excel_path}")
    print(f"Cleaned: {clean_dir}")
    print(f"Mask   : {mask_dir}")