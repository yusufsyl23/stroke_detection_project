#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIRALI (ARDIŞIK) BINARY MASK DUPLICATE TEMİZLEYİCİ
==================================================

Senin veri koşullarına göre optimize edilmiştir:
- Maskeler zaten ROI crop ile 512x512'e getirilmiş (BURADA RESIZE/PADDING YOK!)
- Maskeler zaten binary (BURADA THRESHOLD ile tekrar binary yapmıyoruz!)
- Duplicate'lar sıralı geliyor (02_slice0031, 0032, 0033 gibi) -> O(n) hız

Mantık:
1) Dosya adından patient ve slice parse edilir (örn: 02_slice0031.png)
2) Aynı patient içinde slice numarasına göre sıralanır
3) Yürüyerek (sadece komşu/ardışık) kıyas yapılır:
   - Önce np.array_equal ile "tam aynı" hızlı yakala
   - Değilse IoU >= threshold ise duplicate say (hata payı burada)
4) Her duplicate grup için:
   - İlk dosya KORUNAN
   - Diğerleri SİLİNDİ (yanına IoU ve benzerlik% yazılır)
5) Dry-run rapor + log üretir, sonra "EVET" dersen siler

Çıktılar:
- onizleme_silinenler.txt
- onizleme_detayli_log.txt
- (silme yapılırsa) silinenler.txt, silme_logu.txt
"""

import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("❌ Pillow yok. Kur: pip install Pillow")
    sys.exit(1)

# -----------------------------
# AYARLAR
# -----------------------------
EXPECTED_SHAPE = (512, 512)

# Hata payı için IoU eşiği:
IOU_THRESHOLD = 0.97

# Güvenlik: binary olmayan maskeleri otomatik düzeltme (varsayılan kapalı)
# Sen "maskeler zaten binary" dediğin için KAPALI tutuyoruz.
ALLOW_AUTOFIX_NONBINARY = False

# Boş maskeleri (tam siyah) duplicate kabul et:
# İki maske de boşsa IoU=1.0 sayacağız.
TREAT_ALL_EMPTY_AS_DUPLICATE = True

# Dosya adı parse: 02_slice0031.png -> patient=02, slice=0031
FILENAME_RE = re.compile(r"^(?P<patient>\d+).*?slice(?P<slice>\d+)\.png$", re.IGNORECASE)


def parse_patient_slice(filename: str):
    """
    Dosya adından patient ve slice çıkarır.
    Başarılıysa: (patient_int, slice_int)
    Değilse: None
    """
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    return int(m.group("patient")), int(m.group("slice"))


def read_mask_as_binary(path: Path):
    """
    Maskeyi okur ve güvenli şekilde bool binary maske döndürür.

    Kabul edilen:
    - {0,255}
    - {0,1}

    Eğer binary değilse:
    - ALLOW_AUTOFIX_NONBINARY=True ise threshold ile binary'e çevirir ve uyarı döner
    - False ise mask None döner ve bu dosya SKIP edilir
    """
    try:
        with Image.open(path) as img:
            arr = np.array(img.convert("L"))
    except Exception as e:
        return None, f"okuma_hatasi: {e}"

    # Boyut kontrol (senin pipeline 512x512 dediğin için burada düzeltme yok)
    if arr.shape != EXPECTED_SHAPE:
        return None, f"boyut_farkli: {arr.shape} != {EXPECTED_SHAPE}"

    vals = np.unique(arr)
    vals_set = set(int(v) for v in vals.tolist())

    if vals_set.issubset({0, 255}):
        return (arr == 255), None

    if vals_set.issubset({0, 1}):
        return (arr == 1), None

    # Binary değil -> güvenli yaklaşım: SKIP
    if ALLOW_AUTOFIX_NONBINARY:
        # Not: Sen riskli dediğin için varsayılan kapalı.
        return (arr > 127), f"binary_degildi_autofix: unique={sorted(vals_set)[:10]}{'...' if len(vals_set)>10 else ''}"

    return None, f"binary_degıl: unique={sorted(vals_set)[:10]}{'...' if len(vals_set)>10 else ''}"


def iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """
    a, b: bool maske
    IoU = intersection / union
    """
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        # ikisi de boşsa
        return 1.0 if TREAT_ALL_EMPTY_AS_DUPLICATE else 0.0

    return inter / union


def write_list(path: Path, lines):
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    print("=" * 80)
    print("🧠 SIRALI DUPLICATE TEMİZLEYİCİ (IoU=0.95, güvenli, 512x512)")
    print("=" * 80)

    # Klasör yolu
    if len(sys.argv) > 1:
        directory = sys.argv[1].strip().strip('"').strip("'")
    else:
        directory = input("📁 Mask klasör yolunu gir: ").strip().strip('"').strip("'")

    base = Path(directory)
    if not base.exists():
        print(f"❌ Klasör bulunamadı: {base}")
        sys.exit(1)

    # PNG dosyaları
    pngs = sorted(base.glob("*.png"))
    if not pngs:
        print("❌ Klasörde PNG yok.")
        sys.exit(1)

    # Parse + sırala
    parsed = []
    unparsed = []
    for p in pngs:
        ps = parse_patient_slice(p.name)
        if ps is None:
            unparsed.append(p.name)
        else:
            patient, sl = ps
            parsed.append((patient, sl, p))

    if not parsed:
        print("❌ Dosya adlarından patient/slice parse edilemedi.")
        print("   Beklenen: 02_slice0031.png")
        print("   Regex:", FILENAME_RE.pattern)
        sys.exit(1)

    parsed.sort(key=lambda x: (x[0], x[1]))

    print(f"📌 Toplam PNG: {len(pngs)} | ✅ Parse: {len(parsed)} | ⚠️ Parse edilemeyen: {len(unparsed)}")
    if unparsed:
        print("   İlk 10 parse edilemeyen:", ", ".join(unparsed[:10]))
    print(f"⚙️  IoU Threshold: {IOU_THRESHOLD}")
    print("-" * 80)

    # -----------------------------
    # ANA İŞLEM: yürüyerek kıyasla
    # -----------------------------
    duplicate_groups = []   # her grup: {"keep": Path, "dups": [(Path, iou)]}
    files_to_delete = []
    warnings = []           # skip veya uyarılar

    prev_patient = None
    prev_mask = None
    anchor_path = None      # grubun "KORUNAN" dosyası (anchor)
    current_group = None    # dict

    for patient, sl, path in parsed:
        mask, warn = read_mask_as_binary(path)

        if warn is not None:
            # warn var: ya skip ya da (autofix) devam
            if mask is None:
                warnings.append(f"{path.name}\tSKIP\t{warn}")
                continue
            else:
                warnings.append(f"{path.name}\tUYARI\t{warn}")

        # Yeni patient başladıysa önceki grubu kapat
        if prev_patient is None or patient != prev_patient:
            if current_group and current_group["dups"]:
                duplicate_groups.append(current_group)
                files_to_delete.extend([p for (p, _) in current_group["dups"]])

            # Yeni grubun anchor'ı bu dosya
            anchor_path = path
            prev_mask = mask
            prev_patient = patient
            current_group = {"keep": anchor_path, "dups": []}
            continue

        # Aynı patient içindeyiz: anchor ile kıyasla (anchor = gruptaki ilk dosya)
        # 1) Tam aynı mı?
        if np.array_equal(prev_mask, mask):
            val_iou = 1.0
            is_dup = True
        else:
            # 2) Hata payı için IoU
            val_iou = iou_bool(prev_mask, mask)
            is_dup = (val_iou >= IOU_THRESHOLD)

        if is_dup:
            current_group["dups"].append((path, float(val_iou)))
            # Anchor'ı değiştirmiyoruz: zincirleme yanlış birleşmeyi azaltır
        else:
            # Grup bitti -> kaydet
            if current_group and current_group["dups"]:
                duplicate_groups.append(current_group)
                files_to_delete.extend([p for (p, _) in current_group["dups"]])

            # Yeni grup başlat: yeni anchor bu dosya
            anchor_path = path
            prev_mask = mask
            current_group = {"keep": anchor_path, "dups": []}

        prev_patient = patient

    # Döngü bitti: son grubu kapat
    if current_group and current_group["dups"]:
        duplicate_groups.append(current_group)
        files_to_delete.extend([p for (p, _) in current_group["dups"]])

    # -----------------------------
    # RAPORLAMA (DRY RUN)
    # -----------------------------
    print("\n" + "=" * 80)
    print("📊 DRY RUN ÖZET")
    print("=" * 80)
    print(f"🔢 Duplicate grup sayısı : {len(duplicate_groups)}")
    print(f"🗑️  Silinecek dosya      : {len(files_to_delete)}")
    print(f"⚠️  Uyarı/Skip           : {len(warnings)}")

    # Silinecekler listesi
    preview_list = base / "onizleme_silinenler.txt"
    write_list(preview_list, [p.name for p in files_to_delete])

    # Detaylı log (senin istediğin format + IoU + %)
    preview_log = base / "onizleme_detayli_log.txt"
    log_lines = []
    log_lines.append("SIRALI DUPLICATE TEMİZLEYİCİ - DRY RUN")
    log_lines.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Klasör: {base}")
    log_lines.append(f"Expected shape: {EXPECTED_SHAPE}")
    log_lines.append(f"IoU Threshold: {IOU_THRESHOLD}")
    log_lines.append(f"ALLOW_AUTOFIX_NONBINARY: {ALLOW_AUTOFIX_NONBINARY}")
    log_lines.append(f"TREAT_ALL_EMPTY_AS_DUPLICATE: {TREAT_ALL_EMPTY_AS_DUPLICATE}")
    log_lines.append(f"Duplicate grup sayısı: {len(duplicate_groups)}")
    log_lines.append(f"Silinecek dosya sayısı: {len(files_to_delete)}")
    log_lines.append("=" * 80)
    log_lines.append("")

    # Grup formatı:
    # Grup 2:
    #   KORUNAN: 02_slice0005.png
    #   SİLİNDİ: 02_slice0006.png   (IoU=0.96321 | Benzerlik=%96.32)
    for gi, g in enumerate(duplicate_groups, 1):
        keep = g["keep"].name
        log_lines.append(f"Grup {gi}:")
        log_lines.append(f"  KORUNAN: {keep}")
        for dup_path, val_iou in g["dups"]:
            sim_pct = val_iou * 100.0
            log_lines.append(
                f"  SİLİNDİ: {dup_path.name}   (IoU={val_iou:.5f} | Benzerlik=%{sim_pct:.2f})"
            )
        log_lines.append("")

    if warnings:
        log_lines.append("=" * 80)
        log_lines.append("UYARILAR / SKIP EDİLENLER")
        for w in warnings:
            log_lines.append(w)
        log_lines.append("")

    write_list(preview_log, log_lines)

    print(f"\n📝 Dry-run çıktıları yazıldı:")
    print(f"   - {preview_list.name}")
    print(f"   - {preview_log.name}")

    if not files_to_delete:
        print("\n✨ Silinecek dosya yok. Bitti.")
        return

    # -----------------------------
    # GERÇEK SİLME (kullanıcı onayı)
    # -----------------------------
    ans = input("\n🚀 Gerçek silmeye geçilsin mi? (EVET/hayir): ").strip().upper()
    if ans != "EVET":
        print("❌ Silme iptal. Sadece rapor üretildi.")
        return

    deleted = 0
    for p in files_to_delete:
        try:
            p.unlink()
            deleted += 1
            if deleted % 50 == 0:
                print(f"   Silinen: {deleted}/{len(files_to_delete)}")
        except Exception as e:
            warnings.append(f"{p.name}\tSİLME_HATASI\t{e}")

    deleted_list = base / "silinenler.txt"
    deleted_log = base / "silme_logu.txt"

    write_list(deleted_list, [p.name for p in files_to_delete])

    log2 = []
    log2.append("SIRALI DUPLICATE TEMİZLEYİCİ - GERÇEK SİLME")
    log2.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log2.append(f"Klasör: {base}")
    log2.append(f"IoU Threshold: {IOU_THRESHOLD}")
    log2.append(f"Silinen: {deleted}/{len(files_to_delete)}")
    log2.append("")
    log2.append("Silinen dosyalar:")
    log2.extend([p.name for p in files_to_delete])
    if warnings:
        log2.append("")
        log2.append("UYARILAR:")
        log2.extend(warnings)

    write_list(deleted_log, log2)

    print(f"\n✅ Tamamlandı. Silinen: {deleted}")
    print(f"🧾 Loglar: {deleted_list.name}, {deleted_log.name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ İptal edildi.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)