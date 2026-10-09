import re
from pathlib import Path

# ---------------- DOSYA YOLLARI ----------------
LOG_PATH = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\SiyahVeKaymaTemizMask\onizleme_detayli_log.txt")

OUT_PATH = Path(r"C:\Users\Yusuf Soylu\Desktop\GPT Sil Baştan\Eren Sİyah ve Kaymadan Arındırılmış Veri Seti\silinecek_tekrarlar.txt")
# -----------------------------------------------

LINE_RE = re.compile(
    r"^\s*(?:KORUNAN|SİLİNDİ)\s*:\s*(?P<fname>.+?\.png)\s*(?:\(.+?\))?\s*[Xx]\s*$"
)

def read_text_any_encoding(p: Path) -> str:
    data = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1254", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def main():

    if not LOG_PATH.is_file():
        raise FileNotFoundError(f"Log bulunamadı: {LOG_PATH}")

    text = read_text_any_encoding(LOG_PATH)

    picked = []
    seen = set()

    for line in text.splitlines():

        m = LINE_RE.match(line)
        if not m:
            continue

        fname = m.group("fname").strip()

        if fname not in seen:
            seen.add(fname)
            picked.append(fname)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for name in picked:
            f.write(name + "\n")

    print("Bulunan dosya sayısı:", len(picked))
    print("Kaydedildi:", OUT_PATH)


if __name__ == "__main__":
    main()