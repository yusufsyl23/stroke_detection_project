# Beyin CT — Kafatası Traşlama (Skull Stripping) Pipeline

Beyin BT (CT) dilimlerinden kafatası ve halo piksellerini kaldırıp yalnızca beyin
dokusunu bırakan, klasik görüntü işleme tabanlı bir traşlama hattı. Derin öğrenme
kullanılmaz: histogram pik analizi, ışın yürütme (ray marching), flood fill ve
morfolojik işlemlerle çalışır. Her sürüm, iki uzman okuyucunun manuel maskelerine
karşı Dice / IoU ile ölçülmüştür.

## Pipeline

```
NIfTI CT hacimleri (.nii.gz)
        |
        +-- pencereleme (WL=40, WW=120) + ROI crop  ->  512x512 PNG
        |
        +-- veri temizligi
        |     +-- siyah / bos dilimlerin ayiklanmasi
        |     +-- tekrar eden dilimlerin kaldirilmasi (IoU + pHash hibrit)
        |
        +-- TRASLAMA (V1 -> V5)
        |     +-- histogramda beyin piki ve kafatasi pikinin bulunmasi
        |     +-- t_white esiginin dilim bazinda belirlenmesi
        |     +-- 8 yonlu isin yurutme + erozyon tabanli sinir
        |     +-- geri beslemeli iterasyon (yeterli halo gitmediyse esigi dusur)
        |
        +-- cikti: traslanmis PNG + halo maskesi + istatistik Excel'i
        |
        +-- DEGERLENDIRME: Dice, IoU, mesafe metrikleri (uzman maskelerine karsi)
```

## Traşlama algoritması sürümleri

| Dosya | İşlev |
|---|---|
| [claude traşlama v2.py](claude%20tra%C5%9Flama%20v2.py) | Güvenli ray-marching; sabit eşik penceresi |
| [claude traşlama v3.py](claude%20tra%C5%9Flama%20v3.py) | Veri güdümlü uyarlamalı eşik; `max_depth` 20 px sabit |
| [claude traşlama v4.py](claude%20tra%C5%9Flama%20v4.py) | `fg_pct` değerine göre dinamik `max_depth` (25–147 px), tam histogram taraması, güvenlik kırpması |
| [claude traşlama V4_1.py](claude%20tra%C5%9Flama%20V4_1.py) | V4'ün veri bazlı yeniden kalibrasyonu |
| [claude traşlama v5.py](claude%20tra%C5%9Flama%20v5.py) | Pik seçimi konuma göre bölgelenir (beyin 51–179 HU, kafatası 180–255 HU); `img_p95` tabanlı güvenlik sınırı; geri besleme eşiği düşürür |
| [traslama v4.1 deneme.py](traslama%20v4.1%20deneme.py) | V4.1 üzerinde parametre denemeleri |
| [beyaz alan silme v1.py](beyaz%20alan%20silme%20v1.py) | Parlak (beyaz) alan temizleme, multiprocessing |

Her sürüm bir öncekinde ölçülen somut bir hatayı düzeltir; dosya başlarındaki yorum
blokları hangi hatanın neden ve nasıl giderildiğini satır satır anlatır. Örneğin V3'te
`max_depth` 20 px sabitti, oysa ölçülen kafatası P90 kalınlığı 107 px idi; dilimlerin
%37.8'i kafatası bitmeden kesiliyordu. V4 bunu `fg_pct` tabanlı dinamik derinlikle çözer.

## Ön işleme ve veri hazırlama

| Dosya | İşlev |
|---|---|
| [GPT Crop ile PNG Çevirme.py](GPT%20Crop%20ile%20PNG%20%C3%87evirme.py) | NIfTI → pencerelenmiş + ROI croplu PNG dönüşümü |
| [GPT Crop ile PNG Çevirme Adım Adım.py](GPT%20Crop%20ile%20PNG%20%C3%87evirme%20Ad%C4%B1m%20Ad%C4%B1m.py) | Aynı dönüşümün adım adım / öğretici sürümü |
| [split_raw_data.py](split_raw_data.py) | Makale parametreleriyle (`BINARY_THRES=220`) ham veri ayrıştırma |
| [siyahları_bulma.py](siyahlar%C4%B1_bulma.py) | Boş / siyah dilimlerin tespiti |
| [GPT Tekrar edenleri tasima IoU Phash Hibrit.py](GPT%20Tekrar%20edenleri%20tasima%20IoU%20Phash%20Hibrit.py) | Tekrar eden maskeleri pHash ön filtresi + IoU ile taşıma |
| [GPT Tekarar Eedneleri Silme IoU.py](GPT%20Tekarar%20Eedneleri%20Silme%20IoU.py) | Ardışık dilimlerde IoU tabanlı tekrar silme |
| [tekrar silme dosya bazlı.py](tekrar%20silme%20dosya%20bazl%C4%B1.py) | Metin listesindeki isimlere göre dosya silme |
| [tekrarda X olanları kaydetme.py](tekrarda%20X%20olanlar%C4%B1%20kaydetme.py) | Tekrar raporundan işaretli kayıtların ayıklanması |
| [texten farklı dosyaya taşıma.py](texten%20farkl%C4%B1%20dosyaya%20ta%C5%9F%C4%B1ma.py) | Metin listesine göre dosya taşıma |

## Maske üretimi

| Dosya | İşlev |
|---|---|
| [Binary Maske Çıkartma.py](Binary%20Maske%20%C3%87%C4%B1kartma.py) | Ana makalenin algoritmasıyla JHO veri setinden beyin + binary maske |
| [İstatistiğe Dayalı PNG den Binary Mask Elde etmek.py](%C4%B0statisti%C4%9Fe%20Dayal%C4%B1%20PNG%20den%20Binary%20Mask%20Elde%20etmek.py) | Veri seti istatistiklerine dayalı binary maske üretimi |
| [Sadece Flood Fill ile İstatistiğe Dayalı PNG den Binary Mask Elde etmek.py](Sadece%20Flood%20Fill%20ile%20%C4%B0statisti%C4%9Fe%20Dayal%C4%B1%20PNG%20den%20Binary%20Mask%20Elde%20etmek.py) | Yalnızca flood fill ile maske üretimi (35 hasta, 1253 dilim üzerinden kalibre) |
| [PNG Maskeden Beyin Elde Etme.py](PNG%20Maskeden%20Beyin%20Elde%20Etme.py) | Maske × CT ile beyin görüntüsünün elde edilmesi |
| [Brain mask tıraşlama.py](Brain%20mask%20t%C4%B1ra%C5%9Flama.py) | `mask2 & (~mask1)` — geometriye dokunmadan maske farkı |

## Analiz ve metrikler

| Dosya | İşlev |
|---|---|
| [eren için metrik hesaplama.py](eren%20i%C3%A7in%20metrik%20hesaplama.py) | Dice / IoU / mesafe metrikleri, multiprocessing + Excel raporu |
| [kafatasi_kalinlik_istatistik.py](kafatasi_kalinlik_istatistik.py) | 8 yönlü ışınla gerçek kafatası kalınlığı, histogram pik konumları, önerilen `t_white` |
| [traşlama_istatistik.py](tra%C5%9Flama_istatistik.py) | Traşlama çıktılarının toplu istatistiği |
| [binary maske elde etmek için istatistik hesabı Claude.py](binary%20maske%20elde%20etmek%20i%C3%A7in%20istatistik%20hesab%C4%B1%20Claude.py) | Tüm tahmin PNG'leri için piksel / Otsu / histogram analizi |

## Rapor ve sonuç klasörleri

- [**CT_Metrik_Raporlari/**](CT_Metrik_Raporlari) — ana makale ve alternatif veri seti metrik karşılaştırmaları
- [**Eren Sİyah ve Kaymadan Arındırılmış Veri Seti/**](Eren%20S%C4%B0yah%20ve%20Kaymadan%20Ar%C4%B1nd%C4%B1r%C4%B1lm%C4%B1%C5%9F%20Veri%20Seti)
  - `Metrik Sonuçları/V1…V4_1 Metrikler/` — sürüm başına Dice / IoU sonuçları; her sürüm için
    "sadece traşlama" ve "önce tekrarsız sonra traşlama" varyantları, ayrıca
    `gt küçük 10000` ve "kötüler çıkartıldı" alt kümeleri
  - `Traşlama V1…V5/` — sürüm başına istatistik çıktıları (`ham_veri.csv`, Excel özetleri)
  - `V1…V4 Traslama_Raporu.docx/pdf` ve `Kafatası Kalınlık Raporu.docx` — yazılı raporlar
  - `dataset_stats.xlsx`, `siyah_resimler.txt`, `silinecek_tekrarlar.txt` — veri seti envanteri
- [**PNG data/**](PNG%20data) — `Temiz_Beyin_R2` ve `Temiz_Beyin_v2_Claude` halo histogram / metrik raporları
- [**Manual_Mask_Reader_1/**](Manual_Mask_Reader_1), [**Manual_Mask_Reader_2/**](Manual_Mask_Reader_2) — iki uzman okuyucunun manuel referans maskeleri (NIfTI, 34 + 35 hacim)
- [**mask_pixel_stats.xlsx**](mask_pixel_stats.xlsx) — maske piksel istatistikleri

## Bu repoda olmayanlar

Görüntü veri setleri boyut nedeniyle sürüm kontrolüne alınmamıştır (yaklaşık 1.5 GB,
75.600 PNG). Kod, raporlar, metrik tabloları ve manuel referans maskeleri burada;
PNG ve ham hacim verisi yerel diskte durur:

| Dışarıda bırakılan | Boyut |
|---|---|
| `Eren Sİyah ve Kaymadan Arındırılmış Veri Seti/` altındaki PNG'ler | 847 MB |
| `PNG data/` altındaki PNG'ler | 302 MB |
| `orijinal_CT/` (ham NIfTI hacimleri) | 283 MB |
| `image/`, `label_r2/` | 63 MB |

Kök dizindeki üç PNG ([01_slice0009.png](01_slice0009.png), [01_slice0036.png](01_slice0036.png), [05_slice0026.png](05_slice0026.png)) örnek olarak repoda tutulur.

## Kurulum

```bash
pip install numpy pandas pillow opencv-python scipy nibabel openpyxl matplotlib imagehash
```

## Kullanım

Scriptler komut satırı argümanı almaz; yollar her dosyanın başındaki ayar bloğunda
tanımlıdır. Çalıştırmadan önce `INPUT_DIR` / `OUTPUT_DIR` benzeri sabitleri kendi
dizinlerinize göre düzenleyin:

```python
# ============================================================
# AYARLAR - kendi yollarinla degistir
# ============================================================
INPUT_DIR   = r"...\cleaned_png"
OUTPUT_XLSX = r"...\kafatasi_kalinlik_istatistik_v3.xlsx"
```

Ardından:

```bash
python "claude traşlama v5.py"
```

Traşlama scriptleri çok çekirdekli çalışır (`multiprocessing`) ve tek koşuda
traşlanmış PNG'leri, halo maskelerini ve istatistik Excel'ini birlikte üretir.
