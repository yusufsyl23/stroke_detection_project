#!/usr/bin/env python3
"""
Binary Mask Görüntüleri için IoU Tabanlı Duplicate Temizleyici
Binary (siyah-beyaz) mask görüntüleri için optimize edilmiş
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("❌ HATA: Gerekli kütüphaneler yüklü değil!")
    print("\nLütfen şu komutu çalıştırın:")
    print("pip install Pillow imagehash numpy")
    sys.exit(1)


class IoUDuplicateRemover:
    def __init__(self, directory, iou_threshold=0.95, use_hybrid=True):
        """
        Args:
            directory: PNG dosyalarının bulunduğu klasör
            iou_threshold: IoU eşiği (0-1 arası, yüksek = daha katı)
            use_hybrid: True ise önce pHash ile filtrele, sonra IoU (ÇOK HIZLI)
        """
        self.directory = Path(directory)
        self.iou_threshold = iou_threshold
        self.use_hybrid = use_hybrid
        self.image_data = {}  # {file_path: binary_array}
        self.image_hashes = {}  # pHash için
        self.duplicate_groups = defaultdict(list)
        self.files_to_delete = []

    def load_binary_mask(self, image_path):
        """Binary mask görüntüsünü yükle ve numpy array'e çevir"""
        try:
            with Image.open(image_path) as img:
                # Grayscale'e çevir
                img_gray = img.convert('L')
                # Binary'ye çevir (threshold=127)
                img_array = np.array(img_gray)
                binary_array = (img_array > 127).astype(np.uint8)
                return binary_array
        except Exception as e:
            print(f"⚠️  Uyarı: {image_path.name} yüklenirken hata: {e}")
            return None

    def calculate_iou(self, mask1, mask2):
        """İki binary mask arasındaki IoU'yu hesapla"""
        # Shape kontrolü
        if mask1.shape != mask2.shape:
            # Boyutlar farklıysa, küçük olanı büyük olana resize et
            if mask1.size > mask2.size:
                mask2_resized = np.zeros_like(mask1)
                mask2_resized[:mask2.shape[0], :mask2.shape[1]] = mask2
                mask2 = mask2_resized
            else:
                mask1_resized = np.zeros_like(mask2)
                mask1_resized[:mask1.shape[0], :mask1.shape[1]] = mask1
                mask1 = mask1_resized

        # IoU hesapla
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            # Her iki mask de tamamen siyahsa
            return 1.0 if intersection == 0 else 0.0

        iou = intersection / union
        return iou

    def calculate_hash(self, image_path):
        """pHash hesapla (hybrid mod için)"""
        try:
            with Image.open(image_path) as img:
                return imagehash.phash(img, hash_size=16)
        except Exception as e:
            print(f"⚠️  Uyarı: {image_path.name} hash hesaplanırken hata: {e}")
            return None

    def find_duplicates(self):
        """Duplicate mask'leri bul"""
        print(f"\n🔍 Dosyalar taranıyor: {self.directory}")
        print(f"📊 Ayarlar: IoU threshold={self.iou_threshold}, Hybrid mod={'Açık' if self.use_hybrid else 'Kapalı'}")
        print("-" * 70)

        # Tüm PNG dosyalarını listele
        png_files = list(self.directory.glob("*.png"))
        total_files = len(png_files)

        if total_files == 0:
            print("❌ Bu klasörde PNG dosyası bulunamadı!")
            return

        print(f"📁 Toplam {total_files} PNG dosyası bulundu\n")

        if self.use_hybrid:
            # HYBRID MOD: Önce pHash ile filtrele
            print("🔄 Adım 1/3: pHash ile ön filtreleme...")
            hash_groups = defaultdict(list)

            for idx, file_path in enumerate(png_files, 1):
                if idx % 100 == 0:
                    print(f"   Hash hesaplanan: {idx}/{total_files}")

                file_hash = self.calculate_hash(file_path)
                if file_hash is not None:
                    self.image_hashes[file_path] = file_hash
                    hash_groups[file_hash].append(file_path)

            # Benzer hash'lere sahip adayları bul (threshold=5)
            candidates = []
            processed = set()

            for file1, hash1 in self.image_hashes.items():
                if file1 in processed:
                    continue

                group = [file1]
                processed.add(file1)

                for file2, hash2 in self.image_hashes.items():
                    if file2 in processed:
                        continue

                    if hash1 - hash2 <= 5:  # Hamming distance
                        group.append(file2)
                        processed.add(file2)

                if len(group) > 1:
                    candidates.append(group)

            print(f"✅ {len(candidates)} aday duplicate grup bulundu (pHash)\n")

            # Sadece adaylar için binary mask'leri yükle
            print("🔄 Adım 2/3: Aday mask'ler yükleniyor...")
            files_to_load = set()
            for group in candidates:
                files_to_load.update(group)

            for idx, file_path in enumerate(files_to_load, 1):
                if idx % 50 == 0:
                    print(f"   Yüklenen: {idx}/{len(files_to_load)}")

                mask_data = self.load_binary_mask(file_path)
                if mask_data is not None:
                    self.image_data[file_path] = mask_data

            print(f"✅ {len(self.image_data)} mask yüklendi\n")

            # IoU ile kesin kontrol
            print("🔄 Adım 3/3: IoU ile kesin karşılaştırma...")
            comparison_count = 0

            for group in candidates:
                # Sadece bu grup içinde IoU karşılaştır
                group_files = [f for f in group if f in self.image_data]

                for i in range(len(group_files)):
                    for j in range(i + 1, len(group_files)):
                        file1 = group_files[i]
                        file2 = group_files[j]

                        iou = self.calculate_iou(self.image_data[file1], self.image_data[file2])
                        comparison_count += 1

                        if iou >= self.iou_threshold:
                            # Duplicate bulundu
                            found_group = None
                            for dup_group in self.duplicate_groups.values():
                                if file1 in dup_group or file2 in dup_group:
                                    found_group = dup_group
                                    break

                            if found_group:
                                if file1 not in found_group:
                                    found_group.append(file1)
                                if file2 not in found_group:
                                    found_group.append(file2)
                            else:
                                group_id = len(self.duplicate_groups)
                                self.duplicate_groups[group_id] = [file1, file2]

            print(f"✅ {comparison_count} IoU karşılaştırması yapıldı")

        else:
            # TAM IoU MOD: Tüm dosyaları yükle ve karşılaştır (YAVAŞ!)
            print("🔄 Adım 1/2: Tüm mask'ler yükleniyor...")
            for idx, file_path in enumerate(png_files, 1):
                if idx % 50 == 0:
                    print(f"   Yüklenen: {idx}/{total_files}")

                mask_data = self.load_binary_mask(file_path)
                if mask_data is not None:
                    self.image_data[file_path] = mask_data

            print(f"✅ {len(self.image_data)} mask yüklendi\n")

            # Tüm çiftleri karşılaştır
            print("🔄 Adım 2/2: IoU karşılaştırmaları...")
            files_list = list(self.image_data.keys())
            total_comparisons = len(files_list) * (len(files_list) - 1) // 2
            comparison_count = 0

            print(f"⚠️  Toplam {total_comparisons} karşılaştırma yapılacak (bu uzun sürebilir)")

            for i in range(len(files_list)):
                for j in range(i + 1, len(files_list)):
                    file1 = files_list[i]
                    file2 = files_list[j]

                    iou = self.calculate_iou(self.image_data[file1], self.image_data[file2])
                    comparison_count += 1

                    if comparison_count % 1000 == 0:
                        print(f"   İlerleme: {comparison_count}/{total_comparisons}")

                    if iou >= self.iou_threshold:
                        # Duplicate bulundu
                        found_group = None
                        for dup_group in self.duplicate_groups.values():
                            if file1 in dup_group or file2 in dup_group:
                                found_group = dup_group
                                break

                        if found_group:
                            if file1 not in found_group:
                                found_group.append(file1)
                            if file2 not in found_group:
                                found_group.append(file2)
                        else:
                            group_id = len(self.duplicate_groups)
                            self.duplicate_groups[group_id] = [file1, file2]

            print(f"✅ {comparison_count} IoU karşılaştırması tamamlandı")

        # Silinecek dosyaları belirle
        for group in self.duplicate_groups.values():
            # En eski dosyayı koru
            group.sort(key=lambda x: x.stat().st_mtime)
            self.files_to_delete.extend(group[1:])

        return self.files_to_delete

    def print_summary(self):
        """Özet raporu yazdır"""
        print("\n" + "=" * 70)
        print("📊 ÖZET RAPOR (IoU Tabanlı)")
        print("=" * 70)

        if not self.duplicate_groups:
            print("✨ Harika! Hiç duplicate mask bulunamadı.")
            return

        print(f"🔢 Toplam duplicate grup sayısı: {len(self.duplicate_groups)}")
        print(f"🗑️  Silinecek dosya sayısı: {len(self.files_to_delete)}")
        print(f"💾 Korunacak dosya sayısı: {len(self.duplicate_groups)}")
        print(f"📉 Kazanılacak alan: ~{self._calculate_space_saved()}")

        print("\n📋 DETAYLI GRUP LİSTESİ (İlk 10 grup):")
        print("-" * 70)

        for idx, (group_id, group) in enumerate(list(self.duplicate_groups.items())[:10], 1):
            print(f"\n🔸 Grup {idx} ({len(group)} dosya):")
            for file in group:
                status = "✅ KORUNACAK" if file not in self.files_to_delete else "❌ SİLİNECEK"
                size = file.stat().st_size / 1024
                print(f"   {status:15} {file.name} ({size:.1f} KB)")

        if len(self.duplicate_groups) > 10:
            print(f"\n... ve {len(self.duplicate_groups) - 10} grup daha")

    def _calculate_space_saved(self):
        """Silinecek dosyaların toplam boyutunu hesapla"""
        total_bytes = sum(f.stat().st_size for f in self.files_to_delete)

        if total_bytes < 1024:
            return f"{total_bytes} bytes"
        elif total_bytes < 1024 * 1024:
            return f"{total_bytes / 1024:.2f} KB"
        elif total_bytes < 1024 * 1024 * 1024:
            return f"{total_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"

    def save_deleted_filenames(self, filename="silinen_dosyalar_iou.txt"):
        """Silinecek dosyaların isimlerini kaydet"""
        output_path = self.directory / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for file_path in self.files_to_delete:
                f.write(f"{file_path.name}\n")

        print(f"\n📝 Silinen dosya isimleri kaydedildi: {output_path}")
        return output_path

    def save_deletion_log(self, log_file="silinen_dosyalar_detayli_log_iou.txt"):
        """Detaylı log kaydet"""
        log_path = self.directory / log_file

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"IoU Tabanlı Duplicate Mask Silme Logu\n")
            f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Klasör: {self.directory}\n")
            f.write(f"IoU Threshold: {self.iou_threshold}\n")
            f.write(f"Hybrid Mod: {'Açık' if self.use_hybrid else 'Kapalı'}\n")
            f.write(f"Toplam Silinen: {len(self.files_to_delete)}\n")
            f.write("=" * 70 + "\n\n")

            for idx, group in enumerate(self.duplicate_groups.values(), 1):
                f.write(f"Grup {idx}:\n")
                f.write(f"  KORUNAN: {group[0].name}\n")
                for file in group[1:]:
                    f.write(f"  SİLİNDİ: {file.name}\n")
                f.write("\n")

        print(f"💾 Detaylı log dosyası kaydedildi: {log_path}")
        return log_path

    def delete_duplicates(self):
        """Duplicate'leri gerçekten sil"""
        if not self.files_to_delete:
            print("\n✨ Silinecek dosya yok!")
            return

        # Önce dosya isimlerini kaydet
        print("\n📝 Silinen dosya isimleri kaydediliyor...")
        self.save_deleted_filenames("silinen_dosyalar_iou.txt")

        print("\n⚠️  DİKKAT: Dosyalar kalıcı olarak silinecek!")
        response = input(f"{len(self.files_to_delete)} dosyayı silmek istediğinize emin misiniz? (EVET/hayir): ")

        if response.upper() != "EVET":
            print("❌ İşlem iptal edildi.")
            print("📝 Silinecek dosya isimleri 'silinen_dosyalar_iou.txt' dosyasına kaydedildi.")
            return

        print("\n🗑️  Dosyalar siliniyor...")
        deleted_count = 0

        for file_path in self.files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
                if deleted_count % 50 == 0:
                    print(f"   Silinen: {deleted_count}/{len(self.files_to_delete)}")
            except Exception as e:
                print(f"⚠️  Hata: {file_path.name} silinemedi - {e}")

        print(f"\n✅ İşlem tamamlandı! {deleted_count} dosya silindi.")

        # Detaylı log kaydet
        self.save_deletion_log()


def main():
    print("=" * 70)
    print("🧠 BINARY MASK DUPLICATE TEMIZLEYICI (IoU Tabanlı)")
    print("=" * 70)

    # Klasör yolu
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("\n📁 Mask dosyalarının bulunduğu klasör yolunu girin: ").strip()
        directory = directory.strip('"').strip("'")

    if not os.path.exists(directory):
        print(f"❌ HATA: '{directory}' klasörü bulunamadı!")
        sys.exit(1)

    # Ayarlar
    print("\n⚙️  AYARLAR:")
    print("1. IoU Threshold: 0.95 (0.90-0.99 arası önerilir)")
    print("   • 0.90 = Toleranslı (benzer mask'leri yakalar)")
    print("   • 0.95 = Dengeli (ÖNERILEN)")
    print("   • 0.99 = Çok katı (neredeyse tamamen aynı olmalı)")
    print("\n2. Hybrid Mod: Açık (Önce pHash, sonra IoU - ÇOK HIZLI!)")

    use_defaults = input("\nVarsayılan ayarları kullanmak ister misiniz? (E/h): ").strip().upper()

    if use_defaults == "E" or use_defaults == "":
        iou_threshold = 0.95
        use_hybrid = True
    else:
        try:
            iou_threshold = float(input("IoU threshold (0.90-0.99 arası, önerilen 0.95): ") or "0.95")
            hybrid_choice = input("Hybrid mod kullanılsın mı? (E/h): ").strip().upper()
            use_hybrid = hybrid_choice == "E" or hybrid_choice == ""
        except ValueError:
            print("⚠️  Geçersiz değer, varsayılanlar kullanılıyor.")
            iou_threshold = 0.95
            use_hybrid = True

    # İşlem başlat
    remover = IoUDuplicateRemover(directory, iou_threshold=iou_threshold, use_hybrid=use_hybrid)

    # DRY RUN
    print("\n" + "=" * 70)
    print("🔍 MOD: DRY RUN (Önizleme - Hiçbir dosya silinmeyecek)")
    print("=" * 70)

    remover.find_duplicates()
    remover.print_summary()

    # Gerçek silme
    if remover.files_to_delete:
        print("\n" + "=" * 70)
        response = input("\n🚀 Gerçek silme işlemine geçmek ister misiniz? (E/h): ").strip().upper()

        if response == "E" or response == "":
            remover.delete_duplicates()
        else:
            print("\n💾 Yalnızca rapor oluşturuldu, dosyalar silinmedi.")
            remover.save_deleted_filenames("onizleme_silinen_dosyalar_iou.txt")
            remover.save_deletion_log("onizleme_detayli_raporu_iou.txt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


