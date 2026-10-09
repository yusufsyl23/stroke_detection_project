"""
Veri Seti İstatistik Analizi
=============================
Tüm pred PNG'leri paralel olarak analiz eder.
Her resim için piksel istatistikleri, Otsu threshold değeri,
histogram dağılımı ve boşluk tespiti hesaplanır.
Sonuçlar Excel dosyasına kaydedilir.

Kullanım:
    python analyze_dataset.py --pred_dir ./pred_pngs --output stats.xlsx
    python analyze_dataset.py --pred_dir ./pred_pngs --output stats.xlsx --workers 8
"""

import argparse
import os
import warnings
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.filters import threshold_otsu
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from openpyxl import Workbook
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.series import SeriesLabel

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# TEK RESİM ANALİZ FONKSİYONU (worker)
# ─────────────────────────────────────────────

def analyze_single(png_path: str) -> dict:
    """
    Tek bir pred PNG'yi analiz eder.
    ProcessPoolExecutor ile paralel çalışır — pickle edilebilir olması için
    modül seviyesinde tanımlı olmalı (lambda veya iç fonksiyon OLMAMALI).

    Döndürülen dict anahtarları:
        filename       : dosya adı
        width, height  : görüntü boyutu
        min_val        : minimum piksel değeri
        max_val        : maximum piksel değeri
        mean_val       : ortalama piksel değeri
        std_val        : standart sapma
        median_val     : medyan piksel değeri
        p5, p25, p75, p95 : yüzdelikler
        otsu_threshold : Otsu'nun otomatik bulduğu threshold
        dark_ratio     : piksel < 10 olan piksellerin oranı (% — siyah alan)
        bright_ratio   : piksel > 200 olan piksellerin oranı (% — çok parlak)
        is_empty       : görüntünün %95'inden fazlası siyahsa True (boş slice)
        brain_ratio_at_otsu : Otsu threshold'u uygulandığında beyaz kalan alan %
        hist_0_30      : 0-30 aralığındaki piksel yüzdesi
        hist_30_80     : 30-80 aralığındaki piksel yüzdesi
        hist_80_150    : 80-150 aralığındaki piksel yüzdesi
        hist_150_255   : 150-255 aralığındaki piksel yüzdesi
        unique_values  : kaç farklı piksel değeri var (255'e yuvarlanmış)
        error          : hata varsa mesaj, yoksa ""
    """
    result = {"filename": Path(png_path).name, "error": ""}

    try:
        img = Image.open(png_path).convert("L")
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape
        total = h * w

        result["width"]  = w
        result["height"] = h
        result["min_val"]    = float(arr.min())
        result["max_val"]    = float(arr.max())
        result["mean_val"]   = float(arr.mean())
        result["std_val"]    = float(arr.std())
        result["median_val"] = float(np.median(arr))
        result["p5"]         = float(np.percentile(arr, 5))
        result["p25"]        = float(np.percentile(arr, 25))
        result["p75"]        = float(np.percentile(arr, 75))
        result["p95"]        = float(np.percentile(arr, 95))

        # Otsu threshold — tüm piksel sıfırsa veya tek değerliyse hata verir
        if arr.max() > arr.min():
            try:
                otsu_t = float(threshold_otsu(arr))
            except Exception:
                otsu_t = -1.0
        else:
            otsu_t = -1.0

        result["otsu_threshold"] = otsu_t

        # Karanlık / parlak piksel oranları
        result["dark_ratio"]   = float((arr < 10).sum() / total * 100)
        result["bright_ratio"] = float((arr > 200).sum() / total * 100)

        # Boş slice: piksellerin %95'ten fazlası siyahsa bu slice beyin içermiyor
        result["is_empty"] = bool(result["dark_ratio"] > 95.0)

        # Otsu uygulandığında beyaz kalan alan
        if otsu_t > 0:
            result["brain_ratio_at_otsu"] = float((arr > otsu_t).sum() / total * 100)
        else:
            result["brain_ratio_at_otsu"] = 0.0

        # Histogram bölgeleri (yüzde)
        result["hist_0_30"]    = float(((arr >= 0)   & (arr < 30)).sum()  / total * 100)
        result["hist_30_80"]   = float(((arr >= 30)  & (arr < 80)).sum()  / total * 100)
        result["hist_80_150"]  = float(((arr >= 80)  & (arr < 150)).sum() / total * 100)
        result["hist_150_255"] = float(((arr >= 150) & (arr <= 255)).sum()/ total * 100)

        # Kaç farklı parlaklık seviyesi kullanılmış
        result["unique_values"] = int(len(np.unique(arr.astype(np.uint8))))

    except Exception as e:
        result["error"] = str(e)
        # Hatalı dosya için tüm sayısal alanları -1 yap
        for key in ["width","height","min_val","max_val","mean_val","std_val",
                    "median_val","p5","p25","p75","p95","otsu_threshold",
                    "dark_ratio","bright_ratio","brain_ratio_at_otsu",
                    "hist_0_30","hist_30_80","hist_80_150","hist_150_255",
                    "unique_values"]:
            if key not in result:
                result[key] = -1
        result["is_empty"] = False

    return result


# ─────────────────────────────────────────────
# PARALEL ANALİZ
# ─────────────────────────────────────────────

def analyze_all(png_paths: list, n_workers: int) -> list:
    results = [None] * len(png_paths)
    str_paths = [str(p) for p in png_paths]

    print(f"  {len(png_paths)} dosya, {n_workers} worker ile paralel işleniyor...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(analyze_single, path): i
            for i, path in enumerate(str_paths)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if done % 100 == 0 or done == len(png_paths):
                print(f"  [{done}/{len(png_paths)}] tamamlandı...")

    return results


# ─────────────────────────────────────────────
# EXCEL YAZICI
# ─────────────────────────────────────────────

# Renkler
HDR_FILL    = PatternFill("solid", fgColor="1F3864")   # Koyu lacivert — başlıklar
SUB_FILL    = PatternFill("solid", fgColor="2F5496")   # Orta mavi — alt başlıklar
ALT_FILL    = PatternFill("solid", fgColor="EEF2FF")   # Açık mavi — çift satır
WARN_FILL   = PatternFill("solid", fgColor="FFF3CD")   # Sarı — uyarı
ERR_FILL    = PatternFill("solid", fgColor="F8D7DA")   # Kırmızı — hata/boş
OK_FILL     = PatternFill("solid", fgColor="D4EDDA")   # Yeşil — normal

WHITE_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
HDR_FONT    = Font(name="Arial", bold=True, color="FFFFFF", size=9)
BODY_FONT   = Font(name="Arial", size=9)
BOLD_FONT   = Font(name="Arial", bold=True, size=9)

THIN = Side(style="thin", color="CCCCCC")
THIN_BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT   = Alignment(horizontal="left",   vertical="center")
RIGHT  = Alignment(horizontal="right",  vertical="center")


def _cell(ws, row, col, value=None, font=None, fill=None,
          alignment=None, border=None, number_format=None):
    c = ws.cell(row=row, column=col, value=value)
    if font:         c.font         = font
    if fill:         c.fill         = fill
    if alignment:    c.alignment    = alignment
    if border:       c.border       = border
    if number_format: c.number_format = number_format
    return c


def write_excel(results: list, out_path: str):
    wb = Workbook()

    # ── SHEET 1: RAW DATA ──────────────────────────────────────────────────
    ws_raw = wb.active
    ws_raw.title = "Raw Data"
    ws_raw.freeze_panes = "B3"

    columns = [
        # (başlık, alan adı, genişlik, format)
        ("Dosya Adı",           "filename",              30, "@"),
        ("Genişlik (px)",       "width",                 12, "0"),
        ("Yükseklik (px)",      "height",                12, "0"),
        ("Min Piksel",          "min_val",               12, "0.0"),
        ("Max Piksel",          "max_val",               12, "0.0"),
        ("Ortalama",            "mean_val",              12, "0.00"),
        ("Std Sapma",           "std_val",               12, "0.00"),
        ("Medyan",              "median_val",            12, "0.0"),
        ("P5",                  "p5",                    10, "0.0"),
        ("P25",                 "p25",                   10, "0.0"),
        ("P75",                 "p75",                   10, "0.0"),
        ("P95",                 "p95",                   10, "0.0"),
        ("Otsu Threshold",      "otsu_threshold",        14, "0.0"),
        ("Karanlık Alan %\n(<10)", "dark_ratio",         14, "0.00"),
        ("Parlak Alan %\n(>200)",  "bright_ratio",       14, "0.00"),
        ("Beyin Alanı %\n@Otsu",   "brain_ratio_at_otsu",14, "0.00"),
        ("Boş Slice?",          "is_empty",              10, "@"),
        ("Hist 0-30 %",         "hist_0_30",             12, "0.00"),
        ("Hist 30-80 %",        "hist_30_80",            12, "0.00"),
        ("Hist 80-150 %",       "hist_80_150",           12, "0.00"),
        ("Hist 150-255 %",      "hist_150_255",          12, "0.00"),
        ("Benzersiz Değer",     "unique_values",         14, "0"),
        ("Hata",                "error",                 30, "@"),
    ]

    # Satır 1: Bölüm başlıkları (merge)
    section_headers = [
        (1, 1,  "Dosya Bilgisi",        3),
        (1, 4,  "Piksel İstatistikleri", 9),
        (1, 13, "Threshold Analizi",    4),
        (1, 17, "Durum",               1),
        (1, 18, "Histogram Dağılımı",  4),
        (1, 22, "Diğer",               2),
    ]
    for _, start_col, title, span in section_headers:
        end_col = start_col + span - 1
        ws_raw.merge_cells(start_row=1, start_column=start_col,
                           end_row=1,   end_column=end_col)
        c = ws_raw.cell(row=1, column=start_col, value=title)
        c.font      = WHITE_FONT
        c.fill      = HDR_FILL
        c.alignment = CENTER
        c.border    = THIN_BORDER

    # Satır 2: Sütun başlıkları
    for col_idx, (title, _, width, _) in enumerate(columns, 1):
        c = _cell(ws_raw, 2, col_idx, title,
                  font=HDR_FONT, fill=SUB_FILL,
                  alignment=CENTER, border=THIN_BORDER)
        ws_raw.column_dimensions[get_column_letter(col_idx)].width = width

    ws_raw.row_dimensions[1].height = 22
    ws_raw.row_dimensions[2].height = 32

    # Veri satırları
    for row_idx, res in enumerate(results, 3):
        fill = ERR_FILL if res.get("is_empty") or res.get("error") else (
               ALT_FILL if row_idx % 2 == 0 else None)

        for col_idx, (_, field, _, fmt) in enumerate(columns, 1):
            val = res.get(field, "")
            if isinstance(val, bool):
                val = "EVET" if val else "hayır"
            c = _cell(ws_raw, row_idx, col_idx, val,
                      font=BODY_FONT,
                      fill=fill,
                      alignment=LEFT if col_idx == 1 else RIGHT,
                      border=THIN_BORDER,
                      number_format=fmt)

    # Otsu sütununa renk skalası (koşullu biçimlendirme)
    otsu_col = get_column_letter(13)  # otsu_threshold
    ws_raw.conditional_formatting.add(
        f"{otsu_col}3:{otsu_col}{len(results)+2}",
        ColorScaleRule(
            start_type="min", start_color="63BE7B",   # yeşil — düşük
            mid_type="percentile", mid_value=50, mid_color="FFEB84",  # sarı
            end_type="max", end_color="F8696B"         # kırmızı — yüksek
        )
    )

    # ── SHEET 2: ÖZET İSTATİSTİKLER ───────────────────────────────────────
    ws_sum = wb.create_sheet("Özet İstatistikler")
    ws_sum.freeze_panes = "A2"

    valid = [r for r in results if not r.get("error") and not r.get("is_empty")]
    all_r = [r for r in results if not r.get("error")]

    numeric_fields = [
        ("Otsu Threshold",       "otsu_threshold"),
        ("Ortalama Piksel",      "mean_val"),
        ("Std Sapma",            "std_val"),
        ("Medyan",               "median_val"),
        ("P5",                   "p5"),
        ("P25",                  "p25"),
        ("P75",                  "p75"),
        ("P95",                  "p95"),
        ("Beyin Alanı % @Otsu",  "brain_ratio_at_otsu"),
        ("Karanlık Alan %",      "dark_ratio"),
        ("Parlak Alan %",        "bright_ratio"),
        ("Hist 0-30 %",          "hist_0_30"),
        ("Hist 30-80 %",         "hist_30_80"),
        ("Hist 80-150 %",        "hist_80_150"),
        ("Hist 150-255 %",       "hist_150_255"),
    ]

    sum_headers = ["Metrik", "Min", "Max", "Ortalama", "Std Sapma",
                   "Medyan", "P5", "P25", "P75", "P95",
                   "Önerilen Alt Sınır", "Önerilen Üst Sınır"]

    for col_idx, h in enumerate(sum_headers, 1):
        c = _cell(ws_sum, 1, col_idx, h,
                  font=HDR_FONT, fill=HDR_FILL,
                  alignment=CENTER, border=THIN_BORDER)

    col_widths = [25, 10, 10, 12, 12, 10, 10, 10, 10, 10, 18, 18]
    for i, w in enumerate(col_widths, 1):
        ws_sum.column_dimensions[get_column_letter(i)].width = w
    ws_sum.row_dimensions[1].height = 25

    for row_idx, (label, field) in enumerate(numeric_fields, 2):
        vals = [r[field] for r in valid if r.get(field, -1) >= 0]
        if not vals:
            continue
        arr = np.array(vals)

        mn   = float(arr.min())
        mx   = float(arr.max())
        mean = float(arr.mean())
        std  = float(arr.std())
        med  = float(np.median(arr))
        p5   = float(np.percentile(arr, 5))
        p25  = float(np.percentile(arr, 25))
        p75  = float(np.percentile(arr, 75))
        p95  = float(np.percentile(arr, 95))

        # Otsu için önerilen alt/üst sınır: P5 ve P95 arası (aşırı uçları kes)
        if field == "otsu_threshold":
            lower = round(max(0,   p5  - std * 0.5), 1)
            upper = round(min(255, p95 + std * 0.5), 1)
        else:
            lower = ""
            upper = ""

        fill = ALT_FILL if row_idx % 2 == 0 else None
        for col_idx, val in enumerate(
            [label, mn, mx, mean, std, med, p5, p25, p75, p95, lower, upper], 1
        ):
            fmt = "0.0" if col_idx > 1 and val != "" else "@"
            highlight = OK_FILL if col_idx in (11, 12) and val != "" else fill
            _cell(ws_sum, row_idx, col_idx, val,
                  font=BOLD_FONT if col_idx == 1 else BODY_FONT,
                  fill=highlight,
                  alignment=LEFT if col_idx == 1 else CENTER,
                  border=THIN_BORDER,
                  number_format=fmt)

    # ── SHEET 3: PARAMETRE ÖNERİSİ ────────────────────────────────────────
    ws_par = wb.create_sheet("Parametre Önerisi")

    otsu_vals = [r["otsu_threshold"] for r in valid if r.get("otsu_threshold", -1) > 0]
    brain_vals = [r["brain_ratio_at_otsu"] for r in valid if r.get("brain_ratio_at_otsu", -1) >= 0]

    if otsu_vals:
        ot_arr = np.array(otsu_vals)
        p5_ot  = float(np.percentile(ot_arr, 5))
        p95_ot = float(np.percentile(ot_arr, 95))
        std_ot = float(ot_arr.std())
        sug_lo = round(max(0,   p5_ot  - std_ot * 0.3), 1)
        sug_hi = round(min(255, p95_ot + std_ot * 0.3), 1)
    else:
        sug_lo, sug_hi = 10, 80

    n_empty  = sum(1 for r in results if r.get("is_empty"))
    n_errors = sum(1 for r in results if r.get("error"))
    n_valid  = len(valid)

    rows = [
        ("GENEL BİLGİ", None, None, None),
        ("Toplam Resim",            len(results),  "",   ""),
        ("Geçerli (beyin var)",     n_valid,        "",   ""),
        ("Boş Slice",               n_empty,        "",   "⚠ Bu slicelar için maske tamamen siyah olacak"),
        ("Hatalı Dosya",            n_errors,       "",   "⚠ Kontrol edilmeli"),
        ("", None, None, None),
        ("OTSU THRESHOLD ANALİZİ", None, None, None),
        ("Minimum Otsu",            round(min(otsu_vals),1) if otsu_vals else "-", "", ""),
        ("Maximum Otsu",            round(max(otsu_vals),1) if otsu_vals else "-", "", ""),
        ("Ortalama Otsu",           round(float(np.mean(otsu_vals)),1) if otsu_vals else "-", "", ""),
        ("Std Sapma Otsu",          round(std_ot,1) if otsu_vals else "-", "", ""),
        ("P5 Otsu",                 round(p5_ot,1) if otsu_vals else "-", "", "P5 altındaki değerler nadir — muhtemelen gürültü"),
        ("P95 Otsu",                round(p95_ot,1) if otsu_vals else "-", "", "P95 üstündeki değerler nadir — muhtemelen aşırı parlak slice"),
        ("", None, None, None),
        ("ÖNERİLEN HİBRİT PARAMETRELER", None, None, None),
        ("Önerilen Alt Sınır (min_threshold)", sug_lo, "", "Otsu bu değerin altına düşerse bu değer kullanılır"),
        ("Önerilen Üst Sınır (max_threshold)", sug_hi, "", "Otsu bu değerin üstüne çıkarsa bu değer kullanılır"),
        ("", None, None, None),
        ("NOT", None, None, None),
        ("Bu değerler otomatik hesaplanmıştır.", "", "", ""),
        ("Veri setinize göre ince ayar yapabilirsiniz.", "", "", ""),
        ("Beyin alanı %0 veya %100 çıkan dosyaları kontrol edin.", "", "", ""),
    ]

    ws_par.column_dimensions["A"].width = 40
    ws_par.column_dimensions["B"].width = 18
    ws_par.column_dimensions["C"].width = 5
    ws_par.column_dimensions["D"].width = 60

    for row_idx, (label, val, _, note) in enumerate(rows, 1):
        if val is None:  # Bölüm başlığı
            ws_par.merge_cells(f"A{row_idx}:D{row_idx}")
            c = _cell(ws_par, row_idx, 1, label,
                      font=WHITE_FONT, fill=HDR_FILL,
                      alignment=LEFT, border=THIN_BORDER)
            continue
        if label == "":
            continue

        is_suggested = "Önerilen" in label
        _cell(ws_par, row_idx, 1, label,
              font=BOLD_FONT if is_suggested else BODY_FONT,
              fill=OK_FILL if is_suggested else (ALT_FILL if row_idx % 2 == 0 else None),
              alignment=LEFT, border=THIN_BORDER)
        _cell(ws_par, row_idx, 2, val,
              font=BOLD_FONT if is_suggested else BODY_FONT,
              fill=OK_FILL if is_suggested else (ALT_FILL if row_idx % 2 == 0 else None),
              alignment=CENTER, border=THIN_BORDER,
              number_format="0.0" if isinstance(val, float) else "0")
        _cell(ws_par, row_idx, 4, note,
              font=BODY_FONT,
              fill=WARN_FILL if "⚠" in str(note) else None,
              alignment=LEFT, border=THIN_BORDER)

    # ── SHEET 4: ANOMALİLER ───────────────────────────────────────────────
    ws_anom = wb.create_sheet("Anomaliler")

    anom_headers = ["Dosya Adı", "Sorun", "Otsu", "Beyin %", "Ortalama", "Açıklama"]
    for col_idx, h in enumerate(anom_headers, 1):
        _cell(ws_anom, 1, col_idx, h,
              font=HDR_FONT, fill=HDR_FILL,
              alignment=CENTER, border=THIN_BORDER)

    anom_widths = [30, 20, 12, 12, 12, 50]
    for i, w in enumerate(anom_widths, 1):
        ws_anom.column_dimensions[get_column_letter(i)].width = w
    ws_anom.row_dimensions[1].height = 25

    anom_row = 2
    for res in results:
        issues = []
        if res.get("error"):
            issues.append(("HATA", ERR_FILL, f"Python hatası: {res['error']}"))
        if res.get("is_empty"):
            issues.append(("BOŞ SLICE", WARN_FILL, "Piksellerin >%95'i siyah — beyin yok"))
        ot = res.get("otsu_threshold", -1)
        if 0 < ot < (sug_lo - 5) if otsu_vals else 5:
            issues.append(("DÜŞÜK OTSU", WARN_FILL, f"Otsu={ot:.1f} — düşük kontrast veya gürültülü resim"))
        if 0 < ot > (sug_hi + 5) if otsu_vals else 80:
            issues.append(("YÜKSEK OTSU", WARN_FILL, f"Otsu={ot:.1f} — çok parlak resim"))
        br = res.get("brain_ratio_at_otsu", -1)
        if br > 60:
            issues.append(("GENİŞ BEYİN", WARN_FILL, f"Beyin alanı=%{br:.1f} — threshold çok düşük olabilir"))
        if 0 < br < 3:
            issues.append(("KÜÇÜK BEYİN", WARN_FILL, f"Beyin alanı=%{br:.1f} — threshold çok yüksek olabilir"))

        for issue_label, issue_fill, desc in issues:
            _cell(ws_anom, anom_row, 1, res["filename"],
                  font=BODY_FONT, fill=issue_fill,
                  alignment=LEFT, border=THIN_BORDER)
            _cell(ws_anom, anom_row, 2, issue_label,
                  font=BOLD_FONT, fill=issue_fill,
                  alignment=CENTER, border=THIN_BORDER)
            _cell(ws_anom, anom_row, 3, round(ot,1) if ot >= 0 else "-",
                  font=BODY_FONT, fill=issue_fill,
                  alignment=CENTER, border=THIN_BORDER)
            _cell(ws_anom, anom_row, 4, round(br,2) if br >= 0 else "-",
                  font=BODY_FONT, fill=issue_fill,
                  alignment=CENTER, border=THIN_BORDER)
            _cell(ws_anom, anom_row, 5, round(res.get("mean_val",-1),2),
                  font=BODY_FONT, fill=issue_fill,
                  alignment=CENTER, border=THIN_BORDER)
            _cell(ws_anom, anom_row, 6, desc,
                  font=BODY_FONT, fill=issue_fill,
                  alignment=LEFT, border=THIN_BORDER)
            anom_row += 1

    if anom_row == 2:
        ws_anom.merge_cells("A2:F2")
        _cell(ws_anom, 2, 1, "✓ Anomali tespit edilmedi.",
              font=BOLD_FONT, fill=OK_FILL, alignment=CENTER)

    # ── KAYDET ────────────────────────────────────────────────────────────
    wb.save(out_path)
    print(f"\nExcel kaydedildi: {out_path}")
    print(f"  Sheet 1 — Raw Data         : {len(results)} satır")
    print(f"  Sheet 2 — Özet İstatistikler")
    print(f"  Sheet 3 — Parametre Önerisi : alt={sug_lo}, üst={sug_hi}")
    print(f"  Sheet 4 — Anomaliler        : {anom_row-2} anomali")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pred PNG veri setini analiz et → Excel raporu"
    )
    parser.add_argument("--pred_dir",
                        default=r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\SiyahVeKaymaTemizBrain_Algoritma_Cikti",
                        help="Pred PNG'lerinin bulunduğu klasör")
    parser.add_argument("--output",
                        default=r"C:\Users\Yusuf Soylu\Desktop\dataset_stats.xlsx",
                        help="Çıktı Excel dosyası")
    parser.add_argument("--workers",  type=int,
                        default=min(8, multiprocessing.cpu_count()),
                        help="Paralel worker sayısı (default: CPU sayısı, max 8)")
    args = parser.parse_args()

    pred_path = Path(args.pred_dir)
    if not pred_path.exists():
        print(f"HATA: Klasör bulunamadı: {pred_path}")
        return

    png_files = sorted(
        list(pred_path.glob("*.png")) +
        list(pred_path.glob("*.PNG"))
    )
    if not png_files:
        print(f"HATA: {pred_path} içinde PNG dosyası bulunamadı.")
        return

    print(f"\n{'='*55}")
    print(f"  Pred PNG Veri Seti Analizi")
    print(f"{'='*55}")
    print(f"  Klasör  : {pred_path}")
    print(f"  Dosya   : {len(png_files)} PNG")
    print(f"  Workers : {args.workers}")
    print(f"  Çıktı   : {args.output}")
    print(f"{'='*55}\n")

    results = analyze_all(png_files, args.workers)

    n_ok    = sum(1 for r in results if not r.get("error") and not r.get("is_empty"))
    n_empty = sum(1 for r in results if r.get("is_empty"))
    n_err   = sum(1 for r in results if r.get("error"))

    print(f"\nAnaliz tamamlandı:")
    print(f"  ✓ Geçerli  : {n_ok}")
    print(f"  ○ Boş      : {n_empty}")
    print(f"  ✗ Hatalı   : {n_err}")

    write_excel(results, args.output)


if __name__ == "__main__":
    main()