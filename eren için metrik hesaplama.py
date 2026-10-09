"""
Mask Karşılaştırma ve Analiz Scripti - HIZLI (Multiprocessing)
===============================================================
Optimizasyonlar:
  - ProcessPoolExecutor ile tüm CPU çekirdekleri paralel kullanım
  - ThreadPoolExecutor ile paralel dosya okuma (I/O)
  - NumPy vektörel batch hesaplama
  - Bellek verimli işleme

Gereksinimler:
  pip install pillow pandas scipy openpyxl matplotlib
"""

import os
import re
import warnings
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')

# ── Opsiyonel kütüphaneler ───────────────────────────────────────────────────
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("⚠️  openpyxl yok. Kurmak için: pip install openpyxl")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

from scipy.spatial.distance import directed_hausdorff

# ── Sabitler ─────────────────────────────────────────────────────────────────
CPU_WORKERS = max(1, multiprocessing.cpu_count() - 1)
BATCH_SIZE  = 128   # Her batch kaç çift işlesin


# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR (modül seviyesinde – pickle için zorunlu)
# ══════════════════════════════════════════════════════════════════════════════

def load_mask(filepath: str):
    """Mask dosyasını binary uint8 numpy array olarak yükle."""
    try:
        img = Image.open(filepath).convert('L')
        return (np.asarray(img) > 128).astype(np.uint8)
    except Exception as e:
        print(f"  [HATA] Yüklenemedi: {filepath} – {e}")
        return None


def parse_filename(filename: str):
    """01_slice0009.png → (1, 9)"""
    m = re.match(r'(\d+)_slice(\d+)\.png$', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def compute_metrics_single(args):
    """
    Tek bir mask çifti için tüm metrikleri hesapla.
    args = (patient_no, slice_no, gt_path, pred_path)
    """
    patient_no, slice_no, gt_path, pred_path = args

    gt_mask   = load_mask(gt_path)
    pred_mask = load_mask(pred_path)

    if gt_mask is None or pred_mask is None:
        return None

    if gt_mask.shape != pred_mask.shape:
        return None

    pred_f = pred_mask.flatten()
    gt_f   = gt_mask.flatten()

    TP = int(np.sum((pred_f == 1) & (gt_f == 1)))
    FP = int(np.sum((pred_f == 1) & (gt_f == 0)))
    TN = int(np.sum((pred_f == 0) & (gt_f == 0)))
    FN = int(np.sum((pred_f == 0) & (gt_f == 1)))

    total       = TP + FP + TN + FN
    accuracy    = (TP + TN) / total if total else 0.0
    precision   = TP / (TP + FP)   if (TP + FP)   else 0.0
    recall      = TP / (TP + FN)   if (TP + FN)   else 0.0
    specificity = TN / (TN + FP)   if (TN + FP)   else 0.0
    dice        = 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) else 0.0
    iou         = TP / (TP + FP + FN)     if (TP + FP + FN)   else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0)

    pred_pts = np.argwhere(pred_mask == 1)
    gt_pts   = np.argwhere(gt_mask   == 1)
    if len(pred_pts) > 0 and len(gt_pts) > 0:
        hd = max(directed_hausdorff(pred_pts, gt_pts)[0],
                 directed_hausdorff(gt_pts, pred_pts)[0])
    else:
        hd = float('nan')

    return {
        'Patient_No':         patient_no,
        'Slice_No':           slice_no,
        'GT_File':            os.path.basename(gt_path),
        'Pred_File':          os.path.basename(pred_path),
        'Accuracy':           accuracy,
        'Precision':          precision,
        'Recall':             recall,
        'Specificity':        specificity,
        'F1_Score':           f1,
        'Dice_Score':         dice,
        'IoU':                iou,
        'Hausdorff_Distance': hd,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ANA SINIF
# ══════════════════════════════════════════════════════════════════════════════

class MaskComparatorFast:
    def __init__(self, gt_folder: str, pred_folder: str, output_folder: str = 'results'):
        self.gt_folder     = gt_folder
        self.pred_folder   = pred_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        self.gt_files   = {}
        self.pred_files = {}
        self.results    = []

    # ── Dosya indexleme ───────────────────────────────────────────────────────
    def index_files(self):
        print("Dosyalar indexleniyor...")
        for folder, store, label in [
            (self.gt_folder,   self.gt_files,   "Ground Truth"),
            (self.pred_folder, self.pred_files, "Prediction"),
        ]:
            if not os.path.exists(folder):
                raise FileNotFoundError(f"{label} klasörü bulunamadı: {folder}")
            for fn in os.listdir(folder):
                if fn.endswith('.png'):
                    p, s = parse_filename(fn)
                    if p is not None:
                        store[(p, s)] = os.path.join(folder, fn)
            print(f"  ✓ {label}: {len(store)} dosya")

        common = set(self.gt_files) & set(self.pred_files)
        print(f"  ✓ Eşleşen çift: {len(common)}")
        return common

    # ── Paralel işleme ────────────────────────────────────────────────────────
    def process_files(self, common_keys):
        print(f"\nDosyalar işleniyor...")
        print(f"  CPU çekirdek: {CPU_WORKERS}  |  Toplam çift: {len(common_keys)}")

        # Her iş birimi: (patient_no, slice_no, gt_path, pred_path)
        tasks = [
            (p, s, self.gt_files[(p, s)], self.pred_files[(p, s)])
            for p, s in sorted(common_keys)
        ]

        total     = len(tasks)
        processed = 0
        errors    = 0

        # ProcessPoolExecutor → her çekirdeğe iş dağıt
        with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
            futures = {executor.submit(compute_metrics_single, t): t for t in tasks}

            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    errors += 1
                else:
                    self.results.append(result)
                    processed += 1

                done = processed + errors
                if done % 200 == 0 or done == total:
                    pct = 100 * done / total
                    print(f"  İlerleme: {done}/{total} (%{pct:.1f})")

        print(f"\n✓ Tamamlandı!  Başarılı: {processed}  Hatalı: {errors}")
        return processed, errors

    # ── DataFrame ─────────────────────────────────────────────────────────────
    def create_dataframe(self):
        if not self.results:
            return None
        df = pd.DataFrame(self.results)
        cols = [
            'Patient_No', 'Slice_No', 'GT_File', 'Pred_File',
            'Dice_Score', 'IoU',
            'Accuracy', 'Precision', 'Recall', 'Specificity',
            'F1_Score', 'Hausdorff_Distance',
            'TP', 'FP', 'TN', 'FN',
        ]
        return df[cols]

    # ── Excel ─────────────────────────────────────────────────────────────────
    def save_to_excel(self, df, filename='mask_comparison_results.xlsx'):
        if not EXCEL_AVAILABLE:
            print("Excel çıktısı devre dışı (openpyxl yok).")
            # CSV olarak kaydet
            csv_path = os.path.join(self.output_folder, 'mask_comparison_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"✓ CSV olarak kaydedildi: {csv_path}")
            return csv_path

        filepath = os.path.join(self.output_folder, filename)
        print(f"\nExcel oluşturuluyor: {filename}")

        metric_cols = ['Accuracy', 'Precision', 'Recall', 'Specificity',
                       'F1_Score', 'Dice_Score', 'IoU', 'Hausdorff_Distance']

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Detaylı Sonuçlar', index=False)
            df[metric_cols].describe().to_excel(writer, sheet_name='Özet İstatistikler')

            patient_avg = df.groupby('Patient_No')[
                ['Accuracy', 'Precision', 'Recall', 'Dice_Score', 'IoU', 'Hausdorff_Distance']
            ].agg(['mean', 'std', 'min', 'max'])
            patient_avg.to_excel(writer, sheet_name='Hasta Bazında')

            pd.DataFrame({
                'Metrik':    metric_cols,
                'Ortalama':  [df[c].mean()   for c in metric_cols],
                'Std Sapma': [df[c].std()    for c in metric_cols],
                'Minimum':   [df[c].min()    for c in metric_cols],
                'Maksimum':  [df[c].max()    for c in metric_cols],
                'Medyan':    [df[c].median() for c in metric_cols],
            }).to_excel(writer, sheet_name='Genel Özet', index=False)

        # Başlık formatlama
        wb = openpyxl.load_workbook(filepath)
        hdr_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        hdr_font = Font(bold=True, color="FFFFFF")
        for ws in wb.worksheets:
            for cell in ws[1]:
                if hasattr(cell, 'column_letter'):
                    cell.fill = hdr_fill
                    cell.font = hdr_font
                    cell.alignment = Alignment(horizontal='center')
            for col in ws.columns:
                letters = [c.column_letter for c in col if hasattr(c, 'column_letter')]
                if not letters:
                    continue
                max_len = max(
                    (len(str(c.value)) for c in col
                     if hasattr(c, 'value') and c.value is not None),
                    default=8,
                )
                ws.column_dimensions[letters[0]].width = min(max_len + 2, 50)
        wb.save(filepath)

        print(f"✓ Excel kaydedildi: {filepath}")
        return filepath

    # ── Görselleştirme ────────────────────────────────────────────────────────
    def create_visualizations(self, df):
        if not PLOT_AVAILABLE:
            return
        print("\nGrafikler oluşturuluyor...")
        metrics = ['Accuracy', 'Precision', 'Recall', 'Dice_Score', 'IoU', 'Hausdorff_Distance']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Mask Karşılaştırma Analizi', fontsize=16, fontweight='bold')
        for idx, m in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            ax.hist(df[m].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(df[m].mean(), color='red', linestyle='--', linewidth=2,
                       label=f"Ort: {df[m].mean():.3f}")
            ax.set_title(f'{m} Dağılımı', fontweight='bold')
            ax.set_xlabel(m); ax.set_ylabel('Frekans')
            ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(self.output_folder, 'metric_distributions.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ {out}")

        pa = df.groupby('Patient_No')[['Dice_Score', 'IoU']].mean()
        fig, ax = plt.subplots(figsize=(14, 6))
        x = range(len(pa)); w = 0.35
        ax.bar([i - w/2 for i in x], pa['Dice_Score'], w, label='Dice Score',
               color='steelblue', alpha=0.8)
        ax.bar([i + w/2 for i in x], pa['IoU'], w, label='IoU',
               color='coral', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(pa.index, rotation=45)
        ax.set_title('Hasta Bazında Ortalama Performans', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        out2 = os.path.join(self.output_folder, 'patient_performance.png')
        plt.savefig(out2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ {out2}")

    # ── Özet konsol çıktısı ───────────────────────────────────────────────────
    def print_summary(self, df):
        print("\n" + "=" * 80)
        print("ÖZET İSTATİSTİKLER")
        print("=" * 80)
        metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity',
                   'F1_Score', 'Dice_Score', 'IoU', 'Hausdorff_Distance']
        print(f"\n{'Metrik':<22} {'Ortalama':<12} {'Std Sapma':<12} {'Min':<10} {'Max':<10}")
        print("-" * 80)
        for m in metrics:
            print(f"{m:<22} {df[m].mean():<12.6f} {df[m].std():<12.6f} "
                  f"{df[m].min():<10.6f} {df[m].max():<10.6f}")

        best  = df.loc[df['Dice_Score'].idxmax()]
        worst = df.loc[df['Dice_Score'].idxmin()]
        print(f"\n🏆 En İyi  → Hasta {int(best['Patient_No']):>3}, "
              f"Slice {int(best['Slice_No']):>4} | "
              f"Dice={best['Dice_Score']:.4f}  IoU={best['IoU']:.4f}")
        print(f"💀 En Kötü → Hasta {int(worst['Patient_No']):>3}, "
              f"Slice {int(worst['Slice_No']):>4} | "
              f"Dice={worst['Dice_Score']:.4f}  IoU={worst['IoU']:.4f}")
        print("=" * 80)

    # ── Ana akış ──────────────────────────────────────────────────────────────
    def run(self):
        print("=" * 80)
        print("MASK KARŞILAŞTIRMA – HIZLI MOD (Multiprocessing)")
        print("=" * 80)
        print(f"Zaman      : {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"GT         : {self.gt_folder}")
        print(f"Prediction : {self.pred_folder}")
        print(f"Çıktı      : {self.output_folder}")
        print(f"CPU worker : {CPU_WORKERS} / {multiprocessing.cpu_count()} çekirdek")
        print("=" * 80)

        common = self.index_files()
        if not common:
            print("\n❌ Eşleşen dosya bulunamadı!")
            return None, None

        t0 = datetime.now()
        self.process_files(common)
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"\n⏱  Toplam işlem süresi: {elapsed:.1f} sn  "
              f"({len(self.results)/elapsed:.1f} çift/sn)")

        if not self.results:
            print("\n❌ İşlenebilir sonuç bulunamadı!")
            return None, None

        df = self.create_dataframe()
        self.print_summary(df)
        excel_path = self.save_to_excel(df)

        try:
            self.create_visualizations(df)
        except Exception as e:
            print(f"Grafik hatası: {e}")

        print("\n" + "=" * 80)
        print(f"✅ ANALİZ TAMAMLANDI! ({datetime.now():%Y-%m-%d %H:%M:%S})")
        print("=" * 80)
        return df, excel_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  –  Windows'ta multiprocessing için if __name__ == "__main__" zorunlu!
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── KULLANICI AYARLARI ────────────────────────────────────────────────────
    GT_FOLDER     = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V1\Sadece Traşlama V1\Sadece Traşlama V1 Uzman Mask"
    PRED_FOLDER   = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V1\Sadece Traşlama V1\Algoritma_Brain_Mask_Gürültülü"
    OUTPUT_FOLDER = (r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Metrik Sonuçları\V1 Metrikler\Sadece Traşlama V1 ve Gürültülü Pred\gt küçük 10000 V1"
                     r"1")
    # ─────────────────────────────────────────────────────────────────────────

    comparator = MaskComparatorFast(GT_FOLDER, PRED_FOLDER, OUTPUT_FOLDER)
    try:
        df, excel_path = comparator.run()
        if df is not None:
            print(f"\n📊 Excel : {excel_path}")
            print(f"📁 Çıktı : {OUTPUT_FOLDER}")
    except Exception as e:
        import traceback
        print(f"\n❌ HATA: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()