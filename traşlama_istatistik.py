"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          BEYİN TRAŞLAMA İSTATİSTİK ANALİZİ v2.0                            ║
║                                                                              ║
║  Bu kod BAĞIMSIZ çalışır. Tıraşlama kodunun çıktısını analiz eder.         ║
║                                                                              ║
║  Gerekenler:                                                                 ║
║    1. Orijinal CT görüntüleri       → ORIGINAL_CT_DIR                       ║
║    2. Uzman beyin maskesi           → BRAIN_MASK_DIR                        ║
║    3. Tıraşlama halo maskesi        → HALO_MASK_DIR  (traslama_v2 çıktısı) ║
║    4. Tıraşlanmış görüntü           → CLEANED_DIR    (traslama_v2 çıktısı) ║
║                                                                              ║
║  Çıktılar:                                                                   ║
║    ├── 01_kalite_raporu.xlsx        Ana kalite raporu (5 sayfa)              ║
║    ├── 02_piksel_istatistikleri.xlsx Piksel dağılımı detayları               ║
║    ├── 03_hasta_analizi.xlsx        Hasta bazlı karşılaştırma                ║
║    └── summary_plots/               6 adet özet grafik (PNG)                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# >>> KLASÖR YOLLARINI BURADAN DEĞİŞTİR
# ============================================================================
ORIGINAL_CT_DIR  = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\SiyahVeKaymaTemizBrain"      # Ham CT görüntüleri
BRAIN_MASK_DIR   = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\SiyahVeKaymaTemizMask"          # Uzman beyin maskesi (binary)
HALO_MASK_DIR    = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V4\Önce Tekrarsız Sonra V4 Traşlama\halo_mask_png"  # traslama_v2 çıktısı
CLEANED_DIR      = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V4\Önce Tekrarsız Sonra V4 Traşlama\cleaned_png"   # traslama_v2 çıktısı
OUTPUT_DIR       = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V4\Önce Tekrarsız Sonra V4 Traşlama\Önce Tekrarsız Sonra V4 Traşlama İstatistik"

IMG_EXT          = "*.png"
NUM_WORKERS      = None   # None = cpu_count - 1

# Kalite eşikleri (beyin yenme yüzdesi bazlı)
THRESH_EXCELLENT = 0.5    # < %0.5  → Mükemmel
THRESH_GOOD      = 3.0    # < %3    → İyi
THRESH_MEDIUM    = 10.0   # < %10   → Orta
# >= %10 → Kötü
# == %100 → Kritik

# ============================================================================

import os
import sys
import warnings
import multiprocessing as mp
from pathlib import Path
from functools import partial
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# RENK ve STİL SABITLERI
# ─────────────────────────────────────────────────────────────────────────────
C_NAVY   = "1F3864";  C_BLUE  = "2E75B6";  C_LBLUE = "D6E4F0"
C_GREEN  = "1B5E20";  C_LGRN  = "C8E6C9";  C_LGRN2 = "DCEDC8"
C_RED    = "B71C1C";  C_LRED  = "FFCDD2";  C_ORG   = "E65100"
C_LORG   = "FFE0B2";  C_YEL   = "FFF9C4";  C_GRAY  = "F5F5F5"
C_DGRAY  = "757575";  C_WHITE = "FFFFFF"

def _fill(c): return PatternFill("solid", start_color=c, end_color=c)
def _border():
    s = Side(style="thin", color="BDBDBD")
    return Border(left=s, right=s, top=s, bottom=s)
def _ca(): return Alignment(horizontal="center", vertical="center", wrap_text=True)
def _wf(sz=10): return Font(name="Arial", bold=True, color="FFFFFF", size=sz)
def _bf(sz=10): return Font(name="Arial", bold=True, size=sz)
def _nf(sz=9):  return Font(name="Arial", size=sz)

QUALITY_FILLS = {
    "Mükemmel":               C_LGRN,
    "İyi":                    C_LGRN2,
    "Orta (Az Beyin Yenmiş)": C_YEL,
    "Kötü (Beyin Yenmiş)":    C_LRED,
    "Tıraşlama Yok":          C_GRAY,
    "Kritik (%100)":          C_LRED,
}


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def binarize(arr: np.ndarray, thr: int = 1) -> np.ndarray:
    return (arr >= thr).astype(np.uint8)


def safe_div(a, b, default=0.0):
    return float(a) / float(b) if b > 0 else default


def dice(m1, m2):
    inter = np.sum((m1 > 0) & (m2 > 0))
    total = np.sum(m1 > 0) + np.sum(m2 > 0)
    return safe_div(2 * inter, total)


def iou(m1, m2):
    inter = np.sum((m1 > 0) & (m2 > 0))
    union = np.sum((m1 > 0) | (m2 > 0))
    return safe_div(inter, union)


def pixel_stats(pixels: np.ndarray, prefix: str) -> dict:
    """Piksel dizisi için 20 istatistik hesapla."""
    nan_dict = {f"{prefix}_{k}": np.nan for k in [
        "count","mean","std","min","p5","p10","p25","median",
        "p75","p90","p95","max","range","iqr","skewness",
        "kurtosis","cv","entropy","dark_pct","mid_pct","bright_pct"]}
    if pixels is None or len(pixels) == 0:
        return nan_dict

    p = pixels.astype(np.float32)
    counts, _ = np.histogram(p, bins=256, range=(0, 256))
    probs = counts / (counts.sum() + 1e-12)
    probs_nz = probs[probs > 0]
    entropy = float(-np.sum(probs_nz * np.log2(probs_nz)))
    mean_v = float(np.mean(p))

    return {
        f"{prefix}_count":     int(len(p)),
        f"{prefix}_mean":      round(mean_v, 4),
        f"{prefix}_std":       round(float(np.std(p)), 4),
        f"{prefix}_min":       round(float(np.min(p)), 4),
        f"{prefix}_p5":        round(float(np.percentile(p, 5)),  4),
        f"{prefix}_p10":       round(float(np.percentile(p, 10)), 4),
        f"{prefix}_p25":       round(float(np.percentile(p, 25)), 4),
        f"{prefix}_median":    round(float(np.median(p)), 4),
        f"{prefix}_p75":       round(float(np.percentile(p, 75)), 4),
        f"{prefix}_p90":       round(float(np.percentile(p, 90)), 4),
        f"{prefix}_p95":       round(float(np.percentile(p, 95)), 4),
        f"{prefix}_max":       round(float(np.max(p)), 4),
        f"{prefix}_range":     round(float(np.max(p) - np.min(p)), 4),
        f"{prefix}_iqr":       round(float(np.percentile(p,75) - np.percentile(p,25)), 4),
        f"{prefix}_skewness":  round(float(scipy_stats.skew(p)), 4),
        f"{prefix}_kurtosis":  round(float(scipy_stats.kurtosis(p)), 4),
        f"{prefix}_cv":        round(safe_div(float(np.std(p)), mean_v), 4),
        f"{prefix}_entropy":   round(entropy, 4),
        f"{prefix}_dark_pct":  round(float(np.mean(p < 64) * 100), 4),
        f"{prefix}_mid_pct":   round(float(np.mean((p>=64)&(p<192)) * 100), 4),
        f"{prefix}_bright_pct":round(float(np.mean(p >= 192) * 100), 4),
    }


def quality_label(brain_eaten_pct: float, shave_px: int, dice_val: float) -> str:
    if shave_px == 0 or dice_val < 0.01:
        return "Tıraşlama Yok"
    if brain_eaten_pct >= 100.0:
        return "Kritik (%100)"
    if brain_eaten_pct >= THRESH_MEDIUM:
        return "Kötü (Beyin Yenmiş)"
    if brain_eaten_pct >= THRESH_GOOD:
        return "Orta (Az Beyin Yenmiş)"
    if brain_eaten_pct >= THRESH_EXCELLENT:
        return "İyi"
    return "Mükemmel"


# ─────────────────────────────────────────────────────────────────────────────
# WORKER — tek dilim analizi
# ─────────────────────────────────────────────────────────────────────────────

def analyze_slice(args: tuple) -> dict:
    fname, paths = args
    row = {"filename": fname, "error": None}

    try:
        ct_arr  = load_gray(paths["ct"])   if paths.get("ct")   else None
        bm_arr  = load_gray(paths["bm"])   if paths.get("bm")   else None
        hm_arr  = load_gray(paths["hm"])   if paths.get("hm")   else None
        cl_arr  = load_gray(paths["cl"])   if paths.get("cl")   else None

        TOTAL_PX = 512 * 512

        # ── Binary maskeler ──────────────────────────────────────────────────
        bm_bin = binarize(bm_arr, 128) if bm_arr is not None else None
        hm_bin = binarize(hm_arr,   1) if hm_arr is not None else None

        # ── Alan metrikleri ──────────────────────────────────────────────────
        brain_px   = int(bm_bin.sum())  if bm_bin is not None else np.nan
        shave_px   = int(hm_bin.sum())  if hm_bin is not None else np.nan

        row.update({
            "total_pixels": TOTAL_PX,
            "brain_pixels": brain_px,
            "shave_pixels": shave_px,
            "brain_pct":    round(safe_div(brain_px, TOTAL_PX) * 100, 4),
            "shave_pct":    round(safe_div(shave_px, TOTAL_PX) * 100, 4),
        })

        # ── Örtüşme (tıraşlama × beyin maskesi) ─────────────────────────────
        if bm_bin is not None and hm_bin is not None:
            overlap_bin     = (bm_bin == 1) & (hm_bin == 1)
            outside_bin     = (bm_bin == 0) & (hm_bin == 1)   # beyin dışında tıraşlama
            overlap_px      = int(overlap_bin.sum())
            outside_px      = int(outside_bin.sum())

            brain_eaten_pct    = safe_div(overlap_px, brain_px) * 100
            shave_in_brain_pct = safe_div(overlap_px, shave_px) * 100
            shave_outside_pct  = safe_div(outside_px, shave_px) * 100

            dice_val = dice(bm_bin, hm_bin)
            iou_val  = iou(bm_bin,  hm_bin)
            qlabel   = quality_label(brain_eaten_pct, shave_px, dice_val)

            row.update({
                "overlap_pixels":        overlap_px,
                "outside_pixels":        outside_px,
                "brain_eaten_px":        overlap_px,
                "brain_eaten_pct":       round(brain_eaten_pct,    4),
                "shave_in_brain_pct":    round(shave_in_brain_pct, 4),
                "shave_outside_pct":     round(shave_outside_pct,  4),
                "dice":                  round(dice_val, 6),
                "iou":                   round(iou_val,  6),
                "quality_label":         qlabel,
                "is_critical":           brain_eaten_pct >= 100.0,
                "is_high_risk":          brain_eaten_pct >= 50.0,
                "manual_review_flag":    brain_eaten_pct >= 50.0 or shave_px == brain_px,
            })

            # ── CT üzerinde piksel istatistikleri ────────────────────────────
            if ct_arr is not None:
                # Tüm tıraşlama bölgesi
                row.update(pixel_stats(ct_arr[hm_bin == 1], "shave_ct"))
                # Beyin içine giren tıraşlama
                row.update(pixel_stats(ct_arr[overlap_bin], "eaten_ct"))
                # Beyin dışındaki tıraşlama (doğru bölge)
                row.update(pixel_stats(ct_arr[outside_bin], "outside_ct"))
                # Kalan beyin (tıraşlanmamış beyin)
                remaining = (bm_bin == 1) & (hm_bin == 0)
                row.update(pixel_stats(ct_arr[remaining], "remaining_brain_ct"))
                # Tüm beyin
                row.update(pixel_stats(ct_arr[bm_bin == 1], "full_brain_ct"))

            # ── Tıraşlanmış görüntü üzerinde piksel istatistikleri ───────────
            if cl_arr is not None:
                row.update(pixel_stats(cl_arr[hm_bin == 1], "shave_cl"))
                row.update(pixel_stats(cl_arr[overlap_bin], "eaten_cl"))

    except Exception as e:
        row["error"] = str(e)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# GRAFİKLER
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(df: pd.DataFrame, plots_dir: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Kalite dağılımı ───────────────────────────────────────────────────
    if "quality_label" in df.columns:
        qc = df["quality_label"].value_counts()
        colors = ["#4CAF50","#8BC34A","#FFF176","#EF9A9A","#BDBDBD","#FF5252"]
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            qc.values, labels=qc.index, autopct="%1.1f%%",
            colors=colors[:len(qc)], startangle=90, pctdistance=0.82)
        for at in autotexts: at.set_fontsize(10)
        ax.set_title("Tıraşlama Kalite Dağılımı", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(plots_dir / "01_quality_pie.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── 2. Beyin yenme histogram ─────────────────────────────────────────────
    if "brain_eaten_pct" in df.columns:
        data = df["brain_eaten_pct"].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(data, bins=60, color="#EF5350", edgecolor="white", alpha=0.85)
        axes[0].axvline(data.mean(),   color="navy",  ls="--", lw=2, label=f"Ort: {data.mean():.2f}%")
        axes[0].axvline(data.median(), color="green", ls="--", lw=2, label=f"Med: {data.median():.2f}%")
        axes[0].set_xlabel("Beyin Yenme %"); axes[0].set_ylabel("Dilim Sayısı")
        axes[0].set_title("Beyin Yenme Yüzdesi Dağılımı"); axes[0].legend()
        data_nz = data[data > 0]
        if len(data_nz):
            axes[1].hist(data_nz, bins=60, color="#FF7043", edgecolor="white", alpha=0.85)
            axes[1].set_xlabel("Beyin Yenme % (>0)")
            axes[1].set_title("Sıfır Olmayan Dilimler")
        plt.suptitle("Beyinden Yenme Analizi", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(plots_dir / "02_brain_eaten_hist.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── 3. Dice dağılımı ─────────────────────────────────────────────────────
    if "dice" in df.columns:
        data = df["dice"].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data, bins=60, color="#1976D2", edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="red", ls="--", lw=2, label=f"Ort: {data.mean():.4f}")
        ax.set_xlabel("Dice Skoru"); ax.set_ylabel("Dilim Sayısı")
        ax.set_title("Dice Skoru Dağılımı"); ax.legend()
        plt.tight_layout(); plt.savefig(plots_dir / "03_dice_hist.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── 4. IoU dağılımı ──────────────────────────────────────────────────────
    if "iou" in df.columns:
        data = df["iou"].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data, bins=60, color="#388E3C", edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="red", ls="--", lw=2, label=f"Ort: {data.mean():.4f}")
        ax.set_xlabel("IoU Skoru"); ax.set_ylabel("Dilim Sayısı")
        ax.set_title("IoU Skoru Dağılımı"); ax.legend()
        plt.tight_layout(); plt.savefig(plots_dir / "04_iou_hist.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── 5. CT yoğunluk karşılaştırması ──────────────────────────────────────
    if "shave_ct_mean" in df.columns and "eaten_ct_mean" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(df["shave_ct_mean"].dropna(), bins=50, color="#42A5F5", alpha=0.7, label="Tıraşlanan (tümü)")
        axes[0].hist(df["eaten_ct_mean"].dropna(), bins=50, color="#EF5350", alpha=0.7, label="Yenilen (beyin içi)")
        if "outside_ct_mean" in df.columns:
            axes[0].hist(df["outside_ct_mean"].dropna(), bins=50, color="#66BB6A", alpha=0.7, label="Beyin dışı tıraşlama")
        axes[0].set_xlabel("Ortalama CT Yoğunluğu"); axes[0].set_ylabel("Dilim Sayısı")
        axes[0].set_title("CT Yoğunluk Karşılaştırması"); axes[0].legend()
        axes[1].scatter(df["brain_eaten_pct"], df["shave_ct_mean"], alpha=0.3, s=5, color="#42A5F5")
        axes[1].set_xlabel("Beyin Yenme %"); axes[1].set_ylabel("CT Ort. Yoğunluk")
        axes[1].set_title("Beyin Yenme vs CT Yoğunluğu")
        plt.suptitle("CT Piksel Yoğunluk Analizi", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(plots_dir / "05_ct_intensity.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── 6. Dilim trendi ──────────────────────────────────────────────────────
    if "brain_eaten_pct" in df.columns:
        fig, ax = plt.subplots(figsize=(16, 4))
        x = range(len(df))
        vals = df["brain_eaten_pct"].fillna(0)
        ax.fill_between(x, vals, alpha=0.45, color="crimson", label="Beyin Yenme %")
        ax.plot(x, vals, color="darkred", lw=0.6, alpha=0.7)
        if "manual_review_flag" in df.columns:
            flags = df[df["manual_review_flag"] == True].index
            ax.scatter(flags, df.loc[flags, "brain_eaten_pct"], color="gold", s=20, zorder=5, label="Manuel inceleme")
        ax.set_xlabel("Dilim Sırası"); ax.set_ylabel("Beyin Yenme %")
        ax.set_title("Tüm Dilimlerde Beyin Yenme Trendi"); ax.legend()
        plt.tight_layout(); plt.savefig(plots_dir / "06_eaten_trend.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── 7. Hasta bazlı kutu grafiği ──────────────────────────────────────────
    if "brain_eaten_pct" in df.columns and "patient_id" in df.columns:
        pt_data = [
            df[df["patient_id"] == pid]["brain_eaten_pct"].dropna().values
            for pid in sorted(df["patient_id"].unique())
        ]
        fig, ax = plt.subplots(figsize=(max(12, len(pt_data)*0.5), 6))
        ax.boxplot(pt_data, labels=sorted(df["patient_id"].unique()),
                   patch_artist=True, medianprops=dict(color="red", lw=2))
        ax.set_xlabel("Hasta ID"); ax.set_ylabel("Beyin Yenme %")
        ax.set_title("Hasta Bazlı Beyin Yenme Dağılımı (Box Plot)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(plots_dir / "07_patient_boxplot.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── 8. Korelasyon ısı haritası ───────────────────────────────────────────
    corr_cols = [c for c in ["brain_eaten_pct","dice","iou","shave_pct","brain_pct",
                              "shave_ct_mean","eaten_ct_mean","shave_ct_entropy",
                              "outside_ct_mean"] if c in df.columns]
    if len(corr_cols) >= 4:
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_cols))); ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr_cols, fontsize=9)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title("Metrikler Arası Korelasyon Isı Haritası")
        plt.tight_layout(); plt.savefig(plots_dir / "08_correlation_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"  ✅ {len(list(plots_dir.glob('*.png')))} grafik kaydedildi: {plots_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL YAZICI — 6 SAYFA
# ─────────────────────────────────────────────────────────────────────────────

def _hdr_cell(ws, row, col, text, fill_c=C_NAVY, font_sz=10):
    cell = ws.cell(row=row, column=col, value=text)
    cell.fill = _fill(fill_c); cell.font = _wf(font_sz)
    cell.alignment = _ca(); cell.border = _border()
    return cell

def _data_cell(ws, row, col, value, fill_c=C_WHITE):
    if pd.isna(value): value = ""
    cell = ws.cell(row=row, column=col, value=value)
    cell.fill = _fill(fill_c); cell.font = _nf()
    cell.alignment = _ca(); cell.border = _border()
    return cell

def write_excel_01(wb, df):
    """Özet Dashboard + Kalite Tablosu + Temel İstatistikler"""
    ws = wb.create_sheet("📊 Özet Dashboard")
    ws.sheet_view.showGridLines = False

    # Başlık
    ws.merge_cells("A1:I1")
    c = ws["A1"]
    c.value = f"🧠 Beyin Tıraşlama Kalite Raporu  |  {len(df)} Dilim  |  {datetime.now().strftime('%d.%m.%Y')}"
    c.font = Font(name="Arial", bold=True, size=15, color=C_NAVY)
    c.alignment = _ca(); c.fill = _fill(C_LBLUE)
    ws.row_dimensions[1].height = 35

    # Kalite dağılımı
    row = 3
    for c2, h in enumerate(["Kalite Kategorisi","Adet","%","Beyin Yenme Kriteri","Renk Kodu"], 1):
        _hdr_cell(ws, row, c2, h)
    ws.row_dimensions[row].height = 28
    row += 1

    q_criteria = {
        "Mükemmel":               f"< %{THRESH_EXCELLENT}",
        "İyi":                    f"%{THRESH_EXCELLENT} – %{THRESH_GOOD}",
        "Orta (Az Beyin Yenmiş)": f"%{THRESH_GOOD} – %{THRESH_MEDIUM}",
        "Kötü (Beyin Yenmiş)":    f"> %{THRESH_MEDIUM}",
        "Kritik (%100)":          "= %100",
        "Tıraşlama Yok":          "Dice < 0.01",
    }
    qc = df["quality_label"].value_counts() if "quality_label" in df.columns else pd.Series()
    for qlabel, qcount in qc.items():
        fc = QUALITY_FILLS.get(qlabel, C_WHITE)
        for c2, val in enumerate([qlabel, qcount,
                                   f"{qcount/len(df)*100:.1f}%",
                                   q_criteria.get(qlabel,"—"),
                                   qlabel], 1):
            cell = ws.cell(row=row, column=c2, value=val)
            cell.fill = _fill(fc); cell.font = _bf() if c2==1 else _nf()
            cell.alignment = _ca(); cell.border = _border()
        row += 1

    # Temel istatistikler tablosu
    row += 1
    for c2, h in enumerate(["Metrik","Ortalama","Medyan","Std","Min","P25","P75","P95","Max"], 1):
        _hdr_cell(ws, row, c2, h)
    row += 1

    key_metrics = [
        ("Beyin Alanı (%)",       "brain_pct"),
        ("Tıraşlama Alanı (%)",   "shave_pct"),
        ("Beyin Yenme (%)",       "brain_eaten_pct"),
        ("Tıraşlama→Beyin (%)",   "shave_in_brain_pct"),
        ("Tıraşlama→Dışarı (%)",  "shave_outside_pct"),
        ("Dice Skoru",            "dice"),
        ("IoU Skoru",             "iou"),
        ("CT Ort. (Tıraşlanan)",  "shave_ct_mean"),
        ("CT Ort. (Yenilen)",     "eaten_ct_mean"),
        ("CT Ort. (Dışarı)",      "outside_ct_mean"),
        ("CT Ort. (Kalan Beyin)", "remaining_brain_ct_mean"),
        ("CT Entropi (Tıraşlanan)","shave_ct_entropy"),
    ]
    for i, (label, col) in enumerate(key_metrics):
        s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        fc = C_LBLUE if i % 2 == 0 else C_WHITE
        vals = [label] + [
            f"{fn(s):.4f}" if len(s) else "N/A"
            for fn in [np.mean, np.median, np.std,
                       np.min, lambda x: np.percentile(x,25),
                       lambda x: np.percentile(x,75),
                       lambda x: np.percentile(x,95), np.max]
        ]
        for c2, v in enumerate(vals, 1):
            _data_cell(ws, row, c2, v, fc)
        row += 1

    for c2, w in enumerate([28,12,12,10,10,10,10,10,10], 1):
        ws.column_dimensions[get_column_letter(c2)].width = w


def write_excel_02(wb, df):
    """Dilim detay tablosu — her satır bir dilim"""
    ws = wb.create_sheet("📋 Dilim Detayları")
    ws.freeze_panes = "B2"; ws.sheet_view.showGridLines = False

    cols = [
        ("Dosya Adı",             "filename"),
        ("Kalite",                "quality_label"),
        ("Beyin Px",              "brain_pixels"),
        ("Beyin %",               "brain_pct"),
        ("Tıraşlama Px",          "shave_pixels"),
        ("Tıraşlama %",           "shave_pct"),
        ("Örtüşme Px",            "overlap_pixels"),
        ("Beyin Yenme Px",        "brain_eaten_px"),
        ("Beyin Yenme %",         "brain_eaten_pct"),
        ("Tıraşlama→Beyin %",     "shave_in_brain_pct"),
        ("Tıraşlama→Dışarı %",    "shave_outside_pct"),
        ("Dice",                   "dice"),
        ("IoU",                    "iou"),
        ("Kritik?",               "is_critical"),
        ("Yüksek Risk?",          "is_high_risk"),
        ("Manuel İnceleme?",      "manual_review_flag"),
        ("CT Ort (Tıraşlanan)",   "shave_ct_mean"),
        ("CT Std (Tıraşlanan)",   "shave_ct_std"),
        ("CT Ort (Yenilen)",      "eaten_ct_mean"),
        ("CT Ort (Dışarı)",       "outside_ct_mean"),
        ("CT Ort (Kalan Beyin)",  "remaining_brain_ct_mean"),
        ("CT Entropi (Tıraş)",    "shave_ct_entropy"),
        ("CT Entropi (Yenilen)",  "eaten_ct_entropy"),
        ("Hata",                  "error"),
    ]

    for c2, (label, _) in enumerate(cols, 1):
        _hdr_cell(ws, 1, c2, label)
    ws.row_dimensions[1].height = 28

    for r2, row_data in enumerate(df.itertuples(index=False), 2):
        rd = row_data._asdict()
        alt = r2 % 2 == 0
        for c2, (_, field) in enumerate(cols, 1):
            val = rd.get(field, "")
            if pd.isna(val): val = ""

            if field == "quality_label":
                fc = QUALITY_FILLS.get(str(val), C_WHITE)
            elif field in ("is_critical","manual_review_flag") and val == True:
                fc = C_LRED
            elif field == "is_high_risk" and val == True:
                fc = C_LORG
            else:
                fc = C_LBLUE if alt else C_WHITE

            _data_cell(ws, r2, c2, val, fc)

    # Koşullu renklendirme — beyin yenme
    ep_col = [f for _,f in cols].index("brain_eaten_pct") + 1
    el = get_column_letter(ep_col)
    ws.conditional_formatting.add(
        f"{el}2:{el}{len(df)+1}",
        ColorScaleRule(start_type="min", start_color=C_LGRN,
                       mid_type="percentile", mid_value=50, mid_color=C_YEL,
                       end_type="max", end_color=C_LRED))

    widths = [22,22,10,9,12,9,10,12,13,14,14,8,8,9,12,14,16,14,14,14,20,16,16,20]
    for c2, w in enumerate(widths[:len(cols)], 1):
        ws.column_dimensions[get_column_letter(c2)].width = w


def write_excel_03(wb, df):
    """Piksel istatistikleri detayı"""
    ws = wb.create_sheet("📈 Piksel İstatistikleri")
    ws.sheet_view.showGridLines = False

    stat_keys = ["count","mean","std","min","p5","p10","p25","median",
                 "p75","p90","p95","max","range","iqr","skewness",
                 "kurtosis","cv","entropy","dark_pct","mid_pct","bright_pct"]
    groups = [
        ("CT — Tıraşlanan Bölge (Tümü)", "shave_ct"),
        ("CT — Yenilen Beyin",           "eaten_ct"),
        ("CT — Beyin Dışı Tıraşlama",    "outside_ct"),
        ("CT — Kalan Beyin Dokusu",      "remaining_brain_ct"),
        ("CT — Tüm Beyin",               "full_brain_ct"),
        ("Cleaned — Tıraşlanan",         "shave_cl"),
        ("Cleaned — Yenilen",            "eaten_cl"),
    ]

    cur = 1
    for grp_label, prefix in groups:
        ws.cell(row=cur, column=1, value=grp_label).fill = _fill(C_BLUE)
        ws.cell(row=cur, column=1).font = _wf(11)
        ws.merge_cells(f"A{cur}:C{cur}")
        ws.cell(row=cur, column=1).alignment = _ca()
        cur += 1

        for c2, h in enumerate(["İstatistik","Tüm Dilimler Ort.","Std"], 1):
            _hdr_cell(ws, cur, c2, h, C_BLUE)
        cur += 1

        for i, sk in enumerate(stat_keys):
            col_name = f"{prefix}_{sk}"
            s = df[col_name].dropna() if col_name in df.columns else pd.Series(dtype=float)
            fc = C_LBLUE if i % 2 == 0 else C_WHITE
            _data_cell(ws, cur, 1, sk, fc)
            ws.cell(row=cur, column=1).font = _bf(9)
            _data_cell(ws, cur, 2, round(s.mean(),4) if len(s) else "N/A", fc)
            _data_cell(ws, cur, 3, round(s.std(), 4) if len(s) else "N/A", fc)
            cur += 1
        cur += 2

    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 14


def write_excel_04(wb, df):
    """Hasta bazlı özet analizi"""
    if "patient_id" not in df.columns:
        return

    ws = wb.create_sheet("👤 Hasta Analizi")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "B2"

    pt = df.groupby("patient_id").agg(
        dilim_sayisi        =("filename",       "count"),
        ort_beyin_yenme     =("brain_eaten_pct","mean"),
        med_beyin_yenme     =("brain_eaten_pct","median"),
        maks_beyin_yenme    =("brain_eaten_pct","max"),
        ort_dice            =("dice",           "mean"),
        ort_brain_pct       =("brain_pct",      "mean"),
        kritik_count        =("is_critical",    "sum"),
        kotu_count          =("quality_label",  lambda x: (x=="Kötü (Beyin Yenmiş)").sum()),
        orta_count          =("quality_label",  lambda x: (x=="Orta (Az Beyin Yenmiş)").sum()),
        iyi_count           =("quality_label",  lambda x: (x=="İyi").sum()),
        mukemmel_count      =("quality_label",  lambda x: (x=="Mükemmel").sum()),
        traslamayok_count   =("quality_label",  lambda x: (x=="Tıraşlama Yok").sum()),
        ort_shave_ct_mean   =("shave_ct_mean",  "mean"),
        ort_eaten_ct_mean   =("eaten_ct_mean",  "mean"),
    ).reset_index()

    pt["kotu_oran_pct"] = (pt["kotu_count"] / pt["dilim_sayisi"] * 100).round(1)
    pt = pt.sort_values("ort_beyin_yenme", ascending=False).reset_index(drop=True)

    headers = ["Hasta","Dilim","Ort.Yenme%","Med.Yenme%","Maks.Yenme%",
               "Kritik","Kötü","Orta","İyi","Mükemmel","TıraşlamaYok",
               "KötüOran%","OrtDice","OrtBeyin%","OrtCT(Tıraş)","OrtCT(Yenilen)"]
    fields  = ["patient_id","dilim_sayisi","ort_beyin_yenme","med_beyin_yenme",
               "maks_beyin_yenme","kritik_count","kotu_count","orta_count",
               "iyi_count","mukemmel_count","traslamayok_count","kotu_oran_pct",
               "ort_dice","ort_brain_pct","ort_shave_ct_mean","ort_eaten_ct_mean"]

    for c2, h in enumerate(headers, 1):
        _hdr_cell(ws, 1, c2, h)
    ws.row_dimensions[1].height = 28

    for r2, row_data in enumerate(pt.itertuples(index=False), 2):
        rd = row_data._asdict()
        alt = r2 % 2 == 0
        kotu_oran = rd.get("kotu_oran_pct", 0)
        if kotu_oran >= 30:  row_fc = C_LRED
        elif kotu_oran >= 15: row_fc = C_LORG
        elif kotu_oran >= 5:  row_fc = C_YEL
        else:                  row_fc = C_LGRN if rd.get("ort_beyin_yenme",99) < 1 else (C_LBLUE if alt else C_WHITE)

        for c2, field in enumerate(fields, 1):
            val = rd.get(field, "")
            if pd.isna(val): val = ""
            if isinstance(val, float): val = round(val, 3)
            _data_cell(ws, r2, c2, val, row_fc)

    for c2, w in enumerate([8,8,12,12,12,8,8,8,8,10,14,12,10,12,14,16], 1):
        ws.column_dimensions[get_column_letter(c2)].width = w


def write_excel_05(wb, df):
    """Kritik ve Manuel İnceleme Gerektiren Vakalar"""
    ws = wb.create_sheet("⚠️ Kritik Vakalar")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:H1")
    ws["A1"].value = "⚠️ KRİTİK VAKALAR — Bu dilimleri manuel olarak inceleyiniz"
    ws["A1"].font = Font(name="Arial", bold=True, size=13, color=C_RED)
    ws["A1"].alignment = _ca()

    crit_cols = [
        ("Dosya Adı",           "filename"),
        ("Kalite",              "quality_label"),
        ("Beyin Yenme %",       "brain_eaten_pct"),
        ("Beyin Px",            "brain_pixels"),
        ("Tıraşlama Px",        "shave_pixels"),
        ("Dice",                "dice"),
        ("CT Ort (Yenilen)",    "eaten_ct_mean"),
        ("Açıklama",            "_aciklama"),
    ]

    for c2, (h, _) in enumerate(crit_cols, 1):
        _hdr_cell(ws, 2, c2, h, C_RED)

    sections = [
        ("=== %100 BEYIN YENME (KRİTİK) ===",
         df[df.get("is_critical", pd.Series(False, index=df.index)) == True],
         "brain_pixels == shave_pixels: tıraşlama maskesi beyin maskesiyle AYNI"),
        ("=== %50-99 BEYIN YENME (YÜKSEK RİSK) ===",
         df[(df.get("brain_eaten_pct", pd.Series(0, index=df.index)) >= 50) &
            (df.get("is_critical", pd.Series(False, index=df.index)) == False)],
         "Beyin dokusunun yarısından fazlası tıraşlandı"),
    ]

    cur = 3
    for section_title, section_df, desc in sections:
        if len(section_df) == 0:
            continue
        ws.cell(row=cur, column=1, value=section_title).fill = _fill(C_LRED)
        ws.cell(row=cur, column=1).font = _bf(11)
        ws.merge_cells(f"A{cur}:H{cur}")
        cur += 1

        for r2, row_data in enumerate(section_df.itertuples(index=False)):
            rd = row_data._asdict()
            for c2, (_, field) in enumerate(crit_cols, 1):
                val = rd.get(field, desc) if field != "_aciklama" else desc
                if pd.isna(val): val = ""
                if isinstance(val, float): val = round(val, 4)
                _data_cell(ws, cur, c2, val, C_LRED)
            cur += 1
        cur += 1

    widths = [24,22,14,12,14,10,16,45]
    for c2, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(c2)].width = w


def write_excel_06(wb, df):
    """Ham veri — tüm sütunlar"""
    ws = wb.create_sheet("📑 Ham Veriler")
    ws.freeze_panes = "B2"; ws.sheet_view.showGridLines = False

    for c2, col in enumerate(df.columns, 1):
        _hdr_cell(ws, 1, c2, col)
        ws.column_dimensions[get_column_letter(c2)].width = max(11, len(col)+2)
    ws.row_dimensions[1].height = 28

    for r2, row_data in enumerate(df.itertuples(index=False), 2):
        alt = r2 % 2 == 0
        for c2, val in enumerate(row_data, 1):
            _data_cell(ws, r2, c2, val, C_LBLUE if alt else C_WHITE)


def write_excels(df: pd.DataFrame, output_dir: Path):
    # Excel 1: Kalite raporu
    wb1 = openpyxl.Workbook(); wb1.remove(wb1.active)
    write_excel_01(wb1, df)
    write_excel_02(wb1, df)
    write_excel_04(wb1, df)
    write_excel_05(wb1, df)
    p1 = output_dir / "01_kalite_raporu.xlsx"
    wb1.save(p1); print(f"  ✅ {p1}")

    # Excel 2: Piksel istatistikleri + ham veri
    wb2 = openpyxl.Workbook(); wb2.remove(wb2.active)
    write_excel_03(wb2, df)
    write_excel_06(wb2, df)
    p2 = output_dir / "02_piksel_istatistikleri.xlsx"
    wb2.save(p2); print(f"  ✅ {p2}")


# ─────────────────────────────────────────────────────────────────────────────
# ANA PROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(" 🧠  BEYİN TRAŞLAMA İSTATİSTİK ANALİZİ v2.0")
    print("=" * 70)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Referans klasör: beyin maskesi veya CT
    ref_dir = None
    for d in [BRAIN_MASK_DIR, ORIGINAL_CT_DIR, HALO_MASK_DIR]:
        if d and Path(d).is_dir():
            ref_dir = Path(d); break

    if ref_dir is None:
        print("❌ Hiçbir klasör bulunamadı! Yolları kontrol edin.")
        sys.exit(1)

    all_files = sorted(f.name for f in ref_dir.glob(IMG_EXT))
    print(f"  📁 Bulunan dosya: {len(all_files)}")
    if not all_files:
        sys.exit(1)

    dirs = {
        "ct": Path(ORIGINAL_CT_DIR)  if ORIGINAL_CT_DIR  and Path(ORIGINAL_CT_DIR).is_dir()  else None,
        "bm": Path(BRAIN_MASK_DIR)   if BRAIN_MASK_DIR   and Path(BRAIN_MASK_DIR).is_dir()   else None,
        "hm": Path(HALO_MASK_DIR)    if HALO_MASK_DIR    and Path(HALO_MASK_DIR).is_dir()    else None,
        "cl": Path(CLEANED_DIR)      if CLEANED_DIR      and Path(CLEANED_DIR).is_dir()      else None,
    }
    print("  Klasörler:")
    for k, p in dirs.items():
        print(f"    {k}: {'✅ ' + str(p) if p else '⚠️  Bulunamadı (atlanacak)'}")

    # İş listesi
    tasks = []
    for fname in all_files:
        paths = {}
        for k, d in dirs.items():
            if d:
                p = d / fname
                if p.exists():
                    paths[k] = p
        tasks.append((fname, paths))

    n_workers = NUM_WORKERS or max(1, mp.cpu_count() - 1)
    print(f"\n  ⚙️  {n_workers} CPU çekirdeği ile işleniyor...")

    rows, done = [], 0
    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(analyze_slice, tasks, chunksize=4):
            rows.append(result)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"  ⏳ {done}/{len(tasks)} tamamlandı...")

    df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    df["patient_id"] = df["filename"].str.extract(r"^(\d+)_")

    # Özet
    print(f"\n  📊 Kalite Dağılımı:")
    if "quality_label" in df.columns:
        for label, count in df["quality_label"].value_counts().items():
            print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")

    if "brain_eaten_pct" in df.columns:
        ep = df["brain_eaten_pct"]
        print(f"\n  Beyin Yenme — Medyan: {ep.median():.3f}%  Ort: {ep.mean():.3f}%  Maks: {ep.max():.3f}%")

    if "is_critical" in df.columns:
        crit = df["is_critical"].sum()
        risk = (df.get("is_high_risk", pd.Series(False)) == True).sum()
        print(f"  Kritik (%100): {crit}  |  Yüksek Risk (>%50): {risk}")

    # CSV
    csv_path = output_dir / "ham_veri.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾 CSV: {csv_path}")

    # Grafikler
    print("  🎨 Grafikler oluşturuluyor...")
    generate_plots(df, output_dir / "grafikler")

    # Excel
    print("  📊 Excel dosyaları yazılıyor...")
    write_excels(df, output_dir)

    print("\n" + "=" * 70)
    print(f"  ✅ TAMAMLANDI → {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()