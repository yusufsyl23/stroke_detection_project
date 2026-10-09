#!/usr/bin/env python3
"""
Dosya Listesinden Silme Scripti
Text dosyasındaki isimlere göre dosyaları siler
"""

import os
from pathlib import Path
from datetime import datetime


def delete_files_from_list(directory, filename_list):
    """
    Verilen listedeki dosyaları sil

    Args:
        directory: Dosyaların bulunduğu klasör
        filename_list: Silinecek dosya isimlerinin listesi
    """
    directory = Path(directory)

    print("=" * 70)
    print("🗑️  DOSYA SİLME İŞLEMİ")
    print("=" * 70)
    print(f"📁 Hedef klasör: {directory}")
    print(f"📊 Silinecek dosya sayısı: {len(filename_list)}")
    print("-" * 70)

    # Dosyaları kontrol et
    files_found = []
    files_not_found = []

    print("\n🔍 Dosyalar kontrol ediliyor...")
    for filename in filename_list:
        file_path = directory / filename
        if file_path.exists():
            files_found.append(file_path)
        else:
            files_not_found.append(filename)

    print(f"\n✅ Bulunan dosyalar: {len(files_found)}")
    print(f"❌ Bulunamayan dosyalar: {len(files_not_found)}")

    if files_not_found:
        print(f"\n⚠️  Bulunamayan dosyalar (ilk 10):")
        for i, filename in enumerate(files_not_found[:10], 1):
            print(f"   {i}. {filename}")
        if len(files_not_found) > 10:
            print(f"   ... ve {len(files_not_found) - 10} dosya daha")

    if not files_found:
        print("\n❌ Silinecek dosya bulunamadı!")
        return

    # Toplam boyut hesapla
    total_size = sum(f.stat().st_size for f in files_found)
    if total_size < 1024 * 1024:
        size_str = f"{total_size / 1024:.2f} KB"
    elif total_size < 1024 * 1024 * 1024:
        size_str = f"{total_size / (1024 * 1024):.2f} MB"
    else:
        size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"

    print(f"\n💾 Toplam silinecek boyut: {size_str}")

    # İlk 10 dosyayı göster
    print(f"\n📋 Silinecek dosyalar (ilk 10):")
    for i, file_path in enumerate(files_found[:10], 1):
        size = file_path.stat().st_size / 1024
        print(f"   {i}. {file_path.name} ({size:.1f} KB)")
    if len(files_found) > 10:
        print(f"   ... ve {len(files_found) - 10} dosya daha")

    # Onay al
    print("\n" + "=" * 70)
    print("⚠️  DİKKAT: Bu işlem geri alınamaz!")
    print("⚠️  Dosyalar kalıcı olarak silinecek (çöp kutusuna gitmez)")
    print("=" * 70)

    response = input(f"\n{len(files_found)} dosyayı silmek istediğinize emin misiniz? (EVET/hayir): ")

    if response.upper() != "EVET":
        print("\n❌ İşlem iptal edildi. Dosyalar silinmedi.")
        return

    # Dosyaları sil
    print("\n🗑️  Dosyalar siliniyor...")
    deleted_count = 0
    failed_deletions = []

    for file_path in files_found:
        try:
            file_path.unlink()
            deleted_count += 1
            if deleted_count % 50 == 0:
                print(f"   Silinen: {deleted_count}/{len(files_found)}")
        except Exception as e:
            failed_deletions.append((file_path.name, str(e)))
            print(f"⚠️  Hata: {file_path.name} silinemedi - {e}")

    # Sonuçları göster
    print("\n" + "=" * 70)
    print("✅ İŞLEM TAMAMLANDI!")
    print("=" * 70)
    print(f"✓ Başarıyla silinen: {deleted_count}")
    print(f"✗ Silinemeden: {len(failed_deletions)}")
    print(f"? Bulunamayan: {len(files_not_found)}")

    # Log dosyası oluştur
    log_filename = f"silme_logu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = directory / log_filename

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Dosya Silme Logu\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Klasör: {directory}\n")
        f.write(f"Toplam silinen: {deleted_count}\n")
        f.write(f"Silinemeden: {len(failed_deletions)}\n")
        f.write(f"Bulunamayan: {len(files_not_found)}\n")
        f.write("=" * 70 + "\n\n")

        f.write("SİLİNEN DOSYALAR:\n")
        for file_path in files_found:
            if file_path.name not in [name for name, _ in failed_deletions]:
                f.write(f"✓ {file_path.name}\n")

        if failed_deletions:
            f.write("\n\nSİLİNEMEYEN DOSYALAR:\n")
            for filename, error in failed_deletions:
                f.write(f"✗ {filename}: {error}\n")

        if files_not_found:
            f.write("\n\nBULUNAMAYAN DOSYALAR:\n")
            for filename in files_not_found:
                f.write(f"? {filename}\n")

    print(f"\n💾 Log dosyası kaydedildi: {log_path}")


def main():
    print("=" * 70)
    print("🗑️  DOSYA SİLME ARACI")
    print("=" * 70)

    # Klasör yolunu al
    directory = input("\n📁 Dosyaların bulunduğu klasör yolunu girin: ").strip()
    directory = directory.strip('"').strip("'")

    if not os.path.exists(directory):
        print(f"❌ HATA: '{directory}' klasörü bulunamadı!")
        return

    # Liste dosyasını al
    list_file = input("📄 Silinecek dosya listesi (txt dosyası): ").strip()
    list_file = list_file.strip('"').strip("'")

    if not os.path.exists(list_file):
        print(f"❌ HATA: '{list_file}' dosyası bulunamadı!")
        return

    # Dosya listesini oku
    with open(list_file, 'r', encoding='utf-8') as f:
        filename_list = [line.strip() for line in f if line.strip()]

    # Silme işlemini başlat
    delete_files_from_list(directory, filename_list)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback

        traceback.print_exc()



