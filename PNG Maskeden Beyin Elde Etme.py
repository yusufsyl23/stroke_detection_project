import os
import cv2
import numpy as np
from pathlib import Path

MASK_DIR = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\PNG data\label_r2"
CT_DIR   = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\PNG data\image"
OUT_DIR  = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\PNG data\Uzman_2_Beyin_PNG"

def imread_unicode(path):
    stream = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)

def imwrite_unicode(path, img):
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
        return True
    return False

os.makedirs(OUT_DIR, exist_ok=True)

mask_files = sorted(Path(MASK_DIR).glob("*.png"))
ct_files   = sorted(Path(CT_DIR).glob("*.png"))
ct_dict = {f.name: f for f in ct_files}

matched = 0
skipped = 0
unmatched = []

for mask_path in mask_files:
    fname = mask_path.name

    ct_path = ct_dict.get(fname)
    if ct_path is None:
        unmatched.append(fname)
        continue

    mask = imread_unicode(mask_path)
    ct   = imread_unicode(ct_path)

    if mask is None or ct is None:
        print(f"[HATA] Okunamadı: {fname}")
        skipped += 1
        continue

    if mask.shape != ct.shape:
        print(f"[SKIP] Boyut uyuşmazlığı ({fname}): mask={mask.shape}, ct={ct.shape}")
        skipped += 1
        continue

    binary = (mask > 0).astype(np.uint8) * 255
    masked_ct = cv2.bitwise_and(ct, ct, mask=binary)

    out_path = Path(OUT_DIR) / fname
    ok = imwrite_unicode(out_path, masked_ct)
    if not ok:
        print(f"[HATA] Yazılamadı: {out_path}")
        skipped += 1
        continue

    matched += 1

print(f"\n✅ Tamamlandı! {matched} görüntü işlendi → {OUT_DIR}")
print(f"⏭️  Atlanan: {skipped}")
if unmatched:
    print(f"⚠️  CT'de karşılığı bulunamayan {len(unmatched)} maske var. İlk 10:")
    for u in unmatched[:10]:
        print("  ", u)