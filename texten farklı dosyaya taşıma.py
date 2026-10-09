import re
import shutil
from pathlib import Path

# =========================================================
# BURAYI SEN DOLDUR
# =========================================================
TEXT_FILE = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V1\Sadece Traşlama V1\Genel Hata V1.txt"             # Text dosyası yolu
SOURCE_DIR = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V1\Sadece Traşlama V1\Sadece Traşlama V1 Uzman Mask"             # İçinde resimlerin olduğu klasör
OUTPUT_DIR = r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\Traşlama V1\Sadece Traşlama V1\Çıkarılanlar\gt küçük 2000"      # Kopyalanacak hedef klasör
FILE_EXTENSION = ".png"                        # Aranacak uzantı: .png / .jpg / .jpeg vb.
RECURSIVE_SEARCH = False                        # Alt klasörlere de baksın mı?
# =========================================================


def read_text_any_encoding(file_path: Path) -> str:
    data = file_path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1254", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    return text.lower()


def extract_possible_filenames(text: str, ext: str) -> set:
    ext_escaped = re.escape(ext.lstrip("."))
    pattern = rf'([A-Za-z0-9_\-\. ]+\.{ext_escaped})'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return {m.strip().lower() for m in matches}


def stem_in_text(stem: str, text: str) -> bool:
    pattern = rf'(?<![A-Za-z0-9_]){re.escape(stem.lower())}(?![A-Za-z0-9_])'
    return re.search(pattern, text) is not None


def main():
    text_file = Path(TEXT_FILE)
    source_dir = Path(SOURCE_DIR)
    output_dir = Path(OUTPUT_DIR)
    ext = FILE_EXTENSION.lower().strip()

    if not ext.startswith("."):
        ext = "." + ext

    if not text_file.exists():
        print(f"[HATA] Text dosyası bulunamadı: {text_file}")
        return

    if not source_dir.exists():
        print(f"[HATA] Kaynak klasör bulunamadı: {source_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_text = read_text_any_encoding(text_file)
    text_lower = normalize_text(raw_text)
    explicit_names = extract_possible_filenames(raw_text, ext)

    if RECURSIVE_SEARCH:
        files = list(source_dir.rglob(f"*{ext}"))
    else:
        files = list(source_dir.glob(f"*{ext}"))

    if not files:
        print(f"[UYARI] {source_dir} içinde {ext} uzantılı dosya bulunamadı.")
        return

    moved_count = 0
    skipped_count = 0
    moved_files = []

    for file_path in files:
        file_name = file_path.name.lower()
        file_stem = file_path.stem.lower()

        match_found = False

        if file_name in explicit_names or file_name in text_lower:
            match_found = True
        elif stem_in_text(file_stem, text_lower):
            match_found = True

        if match_found:
            target_path = output_dir / file_path.name

            # Aynı isimli dosya varsa ezmemek için yeni isim ver
            if target_path.exists():
                i = 1
                while True:
                    new_target = output_dir / f"{file_path.stem}_{i}{file_path.suffix}"
                    if not new_target.exists():
                        target_path = new_target
                        break
                    i += 1

            shutil.move(str(file_path), str(target_path))
            moved_count += 1
            moved_files.append(file_path.name)
            print(f"[TAŞINDI] {file_path.name} -> {target_path.name}")
        else:
            skipped_count += 1

    print("\n===== SONUÇ =====")
    print(f"Toplam taranan dosya : {len(files)}")
    print(f"Taşınan dosya        : {moved_count}")
    print(f"Eşleşmeyen dosya     : {skipped_count}")

    if moved_files:
        print("\nTaşınan dosyalar:")
        for name in moved_files:
            print(f" - {name}")


if __name__ == "__main__":
    main()