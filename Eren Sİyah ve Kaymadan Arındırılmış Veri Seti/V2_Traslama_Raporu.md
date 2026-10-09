# V2 Traşlama Algoritması — Teknik Rapor

## 1. Amaç ve Kapsam

V2 algoritması, V1'in üç yapısal kısıtlamasını gidermek amacıyla geliştirilmiş **güvenli ray-marching** tabanlı bir tıraşlama yöntemidir. Temel hedef V1 ile aynıdır: beyin BT görüntülerinde uzman maskeleme sonrası kalan artık kafatası kemiği (halo) piksellerinin temizlenmesi. V2, bu hedefe ulaşmak için üç yeni mekanizma ekler: maksimum soyma derinliği sınırı, ince dilim tespiti ve bağımsız yön maskeleri.

Pipeline içindeki konumu V1 ile aynıdır: pHash + IoU tabanlı duplicate maske eleme adımından **sonra**, GT karşılaştırması ve metrik hesabından **önce**.

---

## 2. V1'den Farklar — Üç Düzeltme

V2, V1 kodunun başında ASCII çerçeve içinde açıkça belgelenmiş üç düzeltme üzerine inşa edilmiştir.

### Düzeltme 1 — Maksimum Soyma Derinliği (`MAX_STRIP_DEPTH`)

V1'de her yönde ray-marching, `t_white` koşulu sağlandığı sürece görüntünün sonuna kadar ilerliyordu; herhangi bir derinlik sınırı bulunmuyordu. V2, her yönde en fazla `MAX_STRIP_DEPTH = 25` piksel silinmesine izin verir; bu sınıra ulaşıldığında ışın durur.

### Düzeltme 2 — İnce Dilim Tespiti (`is_thin_slice`)

V1, tüm dilimlere aynı `t_white` formülünü (`max(200, 0.90 × max)`) uyguluyordu. V2, non-background piksel oranı `%5`'in altındaki dilimleri "ince/uç" dilim olarak sınıflandırır ve bu dilimlerde çok daha yüksek bir eşik kullanır: `t_white = max(245, 0.98 × max)`.

### Düzeltme 3 — Bağımsız Yön Maskeleri

V1'de dört yön tek bir `out` dizisini paylaşıyordu. Bir yönün sildiği piksel (`out[y,x] = 0`), sonraki yönün aynı koordinatı arka plan zannetmesine ve foreground başlangıcını yanlış tespit etmesine yol açıyordu. V2'de her yön kendi bağımsız maskesine yazar; tüm yönler orijinal görüntüyü (`img_u8`) okur ve hiçbir yön diğerini etkilemez. İşlem sonunda dört maske bitwise OR ile birleştirilir.

---

## 3. Girdi ve Çıktı

| Öğe | Değer |
|---|---|
| Girdi klasörü | `SiyahVeKaymaTemizBrainMaskUyg/` |
| Çıktı klasörü | `Sadece Traşlama V2 Traşlama/` |
| Görüntü formatı | PNG, 8-bit grayscale, 512×512 |
| Çıktı 1 | `cleaned_png/` — tıraşlanmış görüntüler |
| Çıktı 2 | `halo_mask_png/` — silinen halo bölge maskeleri |
| Çıktı 3 | `traslama_v2_istatistik.xlsx` — dilim başına istatistik tablosu |

---

## 4. Global Parametreler

```python
MAX_STRIP_DEPTH      = 25     # Her yönde maksimum soyma derinliği (piksel)
MAX_T_WHITE_FLOOR    = 220    # Normal dilim t_white taban değeri  (V1: 200)
SAFETY_FACTOR        = 0.92   # Normal dilim t_white çarpanı       (V1: 0.90)
THIN_SLICE_THRESHOLD = 0.05   # İnce dilim non-bg piksel oran eşiği (%5)
```

Tüm parametreler modül seviyesinde tanımlıdır ve çalışma anında görüntüye göre değişmez. `MAX_T_WHITE_FLOOR` ve `SAFETY_FACTOR`, V1'e kıyasla biraz daha temkinli değerlere ayarlanmıştır: taban +20 piksel, çarpan +0.02.

---

## 5. Eşik Hesaplama (`auto_thresholds`)

### 5.1. Arka Plan Eşiği (`t_bg`)

```python
border = np.concatenate([
    img_u8[:5,  :].ravel(),    # üst 5 satır
    img_u8[-5:, :].ravel(),    # alt 5 satır
    img_u8[:,  :5].ravel(),    # sol 5 sütun
    img_u8[:, -5:].ravel()     # sağ 5 sütun
])
t_bg = int(np.percentile(border, 99.9))
```

V1 ile aynı yöntem. Görüntünün dört kenarından 5 piksel kalınlığında çerçeve örneklenir (toplam 10.240 piksel). BT görüntülerinde beyin daima merkezde konumlandığı için kenar piksellerin neredeyse tamamı hava (arka plan) olarak kabul edilir. P99.9 alınarak kenardaki nadir gürültü piksellerinin eşiği bozması engellenir. Analiz edilen 1253 slice'ın tamamında `t_bg = 0` çıkmıştır; dataset kenarlarında hiç gürültü bulunmamaktadır.

### 5.2. Non-Background Piksel Oranı

```python
non_bg_pct = float((img_u8 > t_bg).sum()) / img_u8.size
```

**Bu hesaplama V1'de yoktu.** Görüntünün yüzde kaçının arka plan olmadığını ölçer. Beyin gövdesini içeren orta dilimlerde bu oran tipik olarak %15–25 aralığında, ince uç dilimlerde %1–4 aralığında çıkar. Dataset genelinde ölçülen değerler: minimum %0.12, maksimum %36.05, ortalama %18.48, medyan %19.73.

### 5.3. İnce Dilim Tespiti ve `t_white` Hesaplama

```python
if non_bg_pct < THIN_SLICE_THRESHOLD:      # %5'ten az
    t_white = int(max(245, 0.98 * mx))
    is_thin = True
else:
    t_white = int(max(MAX_T_WHITE_FLOOR, SAFETY_FACTOR * mx))
    is_thin = False
```

**İnce dilim kolu:** `t_white = max(245, 0.98 × max)`. `mx = 255` için `t_white = 249`. Yalnızca 249 ve üzeri piksel silinir; bu, son derece yüksek bir eşiktir.

**Normal dilim kolu:** `t_white = max(220, 0.92 × max)`. `mx = 255` için `t_white = 234`.

| Dilim tipi | `t_white` (mx=255) | V1 karşılığı |
|---|---|---|
| Normal | 234 | 229 |
| İnce | 249 | 229 (ayrım yoktu) |

Dataset sonuçları: 136 slice (%10.9) ince dilim olarak sınıflandırılmış, 1117 slice (%89.1) normal dilim olarak işlenmiştir.

---

## 6. Ray-Marching Tıraşlama (`strip_halo`)

### 6.1. Veri Yapıları

```python
mask_L = np.zeros((h, w), dtype=np.uint8)  # Sol → Sağ
mask_R = np.zeros((h, w), dtype=np.uint8)  # Sağ → Sol
mask_T = np.zeros((h, w), dtype=np.uint8)  # Yukarı → Aşağı
mask_B = np.zeros((h, w), dtype=np.uint8)  # Aşağı → Yukarı
```

V1'in tek `out` dizisinin yerini dört bağımsız maske almıştır. Hiçbir yön `img_u8` orijinalini değiştirmez; tüm yönler aynı kaynak görüntüyü okur.

### 6.2. Tarama Mantığı — Sol → Sağ Örneği

```python
for y in range(h):
    x = 0
    while x < w and img_u8[y, x] <= t_bg:
        x += 1
    depth = 0
    while x < w and img_u8[y, x] >= t_white and depth < MAX_STRIP_DEPTH:
        mask_L[y, x] = 255
        x += 1
        depth += 1
```

Her satır için iki faz:

**Faz 1 — Arka plan atlama:** `img_u8[y,x] <= t_bg` olduğu sürece ilerle. Foreground'un başladığı noktaya ulaşınca dur.

**Faz 2 — Kafatası silme:** Üç koşul eş zamanlı sağlanmalıdır:
- `x < w` → görüntü sınırı aşılmamış
- `img_u8[y,x] >= t_white` → hâlâ parlak piksel
- `depth < MAX_STRIP_DEPTH` → 25 piksel sınırına ulaşılmamış

Herhangi bir koşul bozulduğunda Faz 2 sona erer ve geri dönüş yoktur. `mask_L[y,x] = 255` ile işaretleme yapılır; `img_u8` değiştirilmez.

Diğer üç yön (Sağ→Sol, Yukarı→Aşağı, Aşağı→Yukarı) özdeş mantıkla çalışır; yalnızca başlangıç koordinatı ve adım yönü farklıdır.

### 6.3. Birleştirme ve Uygulama

```python
halo_mask = ((mask_L | mask_R | mask_T | mask_B) > 0).astype(np.uint8) * 255
cleaned   = img_u8.copy()
cleaned[halo_mask > 0] = 0
```

Dört maske bitwise OR ile birleştirilir: bir piksel herhangi bir yönden işaretlenmişse final maskede 255 değerini alır. Ardından orijinal görüntünün kopyasında maskeli pikseller sıfırlanır. V1'de silme işlemi tarama anında gerçekleşiyordu; V2'de önce tüm taramalar tamamlanır, ardından tek seferde silme yapılır.

### 6.4. Yön İstatistikleri

```python
dir_pixels = {
    'left':   int((mask_L > 0).sum()),
    'right':  int((mask_R > 0).sum()),
    'top':    int((mask_T > 0).sum()),
    'bottom': int((mask_B > 0).sum()),
}
```

Bağımsız maskeler sayesinde her yönün ne kadar piksel sildiği ayrı ayrı ölçülebilir. V1'de tek `out` dizisi kullanıldığı için bu ayrıştırma mümkün değildi.

---

## 7. Çıktı İstatistikleri

V1'e kıyasla genişletilmiş istatistik tablosu:

| Alan | Açıklama | V1'de var mı? |
|---|---|---|
| `filename` | Dilim dosya adı | ✅ |
| `height`, `width` | Görüntü boyutları | ✅ |
| `img_min/max/mean/std` | Temel piksel istatistikleri | ✅ |
| `img_p95`, `img_p99` | 95. ve 99. yüzdelik dilim | ✅ |
| `t_bg`, `t_white` | Hesaplanan eşikler | ✅ |
| `is_thin_slice` | İnce dilim sınıfı | ❌ |
| `non_bg_pct` | Non-background piksel oranı (%) | ❌ |
| `halo_pixels_removed` | Silinen piksel sayısı | ✅ |
| `halo_pct` | Silinen piksel yüzdesi | ❌ |
| `dir_left/right/top/bottom_px` | Yön bazlı silme sayıları | ❌ |
| `dir_dominant` | En fazla silme yapan yön | ❌ |
| `dir_imbalance` | Yön asimetri katsayısı (0.25–1.0) | ❌ |
| `error` | Hata mesajı | ✅ |

---

## 8. Dataset Analizi — 1253 Slice Sonuçları

### 8.1. Genel Özet

| Metrik | Değer |
|---|---|
| Toplam slice | 1253 |
| Hatalı slice | 0 |
| İnce dilim (thin=True) | 136 (%10.9) |
| Normal dilim (thin=False) | 1117 (%89.1) |
| Hiç silinmeyen slice | 32 |
| Ortalama halo px/slice | 584.7 |
| Medyan halo px/slice | 143.0 |
| Maksimum halo px | 11.924 (11_slice0052) |

Medyan (143) ile ortalama (584) arasındaki büyük fark, dağılımın sağa çarpık olduğuna işaret eder: büyük çoğunluk az silerken az sayıda slice çok yüksek halo miktarına sahiptir.

### 8.2. t_white Dağılımı

| t_white | Slice sayısı | Açıklama |
|---|---|---|
| 220 | 19 | img_max düşük, normal dilim taban devreye girdi |
| 224–230 | 4 | img_max düşük, edge case |
| 234 | 1094 | Normal dilim standart değer (mx=255) |
| 245 | 17 | İnce dilim, img_max < 255, taban devreye girdi |
| 246–249 | 119 | İnce dilim standart değer |

Slice'ların büyük çoğunluğu (%87.3) `t_white=234` ile işlenmiştir.

### 8.3. t_white=220 Grubu — Edge Case

19 slice'ta `img_max` değeri düşük olduğu için (`max(220, 0.92 × max)` hesabında `0.92 × max < 220`) taban değeri devreye girmiştir. Bu grupta `img_max` ortalaması 210'dur. 13 slice'ta hiç silme gerçekleşmemiştir; bu slice'larda görüntünün maksimum değeri `t_white` eşiğinin altında kalmaktadır.

### 8.4. Hiç Silinmeyen 32 Slice

Bu grup iki alt kategoriye ayrılır:

**İnce dilim, img_max < 245:** `t_white=245` hesaplanmış, ancak görüntünün hiçbir pikseli bu eşiği geçememiştir. Örneğin `12_slice0007`: `img_max=155`, `t_white=245`. Bu slice'ta yalnızca iki küçük gri leke bulunmakta, hiç beyaz piksel yer almamaktadır. Algoritma doğru davranmıştır.

**Normal dilim, img_max=255, halo_px=0:** `14_slice0032`, `26_slice0026`, `32_slice0029–30` gibi slice'lar bu gruba girmektedir. `img_max=255` olmasına karşın hiç silme gerçekleşmemiştir. Bu slice'larda görüntünün merkezinde izole parlak piksel (kalsifikasyon, damar) bulunmakta, ancak kenardan başlayan ray-marching bu piksellere Faz 2'ye geçmeden ulaşamamaktadır. Kenar → arka plan → doğrudan beyin geçişi söz konusudur; arada kafatası kemiği bandı yoktur. Algoritma doğru davranmıştır.

### 8.5. Yön Analizi

| Yön | Baskın olduğu slice sayısı |
|---|---|
| Sol | 392 |
| Sağ | 391 |
| Üst | 252 |
| Alt | 186 |
| Hiçbiri (halo=0) | 32 |

Sol-sağ dağılımı neredeyse eşittir; kafatasının sağ-sol simetrisi beklenen bir sonuçtur. Üst-alt eksende daha az baskın yön gözlemlenmektedir: BT görüntüsünde beyin üstte daha dar, altta daha geniş olduğundan üst-alt kafatası halosunun sol-sağa kıyasla daha ince kaldığı anlaşılmaktadır.

Ortalama `dir_imbalance = 0.375` (mükemmel simetri = 0.25, tek yön baskın = 1.0). Bu değer algoritmanın dört yönde dengeli çalıştığını göstermektedir. `dir_imbalance = 1.0` olan yalnızca 5 slice bulunmaktadır; bu slice'larda silme miktarı 1–3 piksel ile ihmal edilebilir düzeydedir.

### 8.6. Hasta Bazlı Özet

35 hasta üzerinden değerlendirildiğinde, en yüksek ortalama halo silme miktarına sahip hastalar Hasta 08 (1901 px/slice), Hasta 11 (1725 px/slice) ve Hasta 10 (1670 px/slice) olarak öne çıkmaktadır. En düşük ortalama ise Hasta 30 (36 px/slice) ve Hasta 26 (71 px/slice) ile ölçülmüştür. Bu farklılıklar, dataset içindeki kafatası kalınlığı ve halo yoğunluğunun hastadan hastaya belirgin biçimde değiştiğini ortaya koymaktadır.

---

## 9. V2'nin Yapısal Sorunları

Görsel karşılaştırma ve dataset analizi, V2'de iki temel sorun ortaya çıkarmıştır.

### 9.1. `MAX_STRIP_DEPTH=25` Yetersizliği

En kritik sorun budur. Kalın halosu olan slice'larda ray-marching 25 piksel sonra durur ve halonun geri kalanını temizleyemez; görüntüde beyaz artık kalır. `10_slice0049` bu sorunun somut örneğidir: V2 çıktısında kalan beyaz piksellerin değeri 255'tir, yani `t_white=249` eşiğini geçmektedir ve silinebilir olmalıdır. Ancak bu pikseller kenardan 25 adımdan daha uzakta konumlandığı için ray-marching oraya ulaşamamaktadır. V1 bu slice'ta derinlik sınırı olmaksızın halonun tamamını temizlemiştir.

### 9.2. `is_thin_slice` Mekanizmasının Sınırlılığı

İnce dilim mekanizması, `non_bg_pct < %5` koşulunu **içerik körü** biçimde değerlendirir: non-background piksellerin beyin dokusu mu yoksa kafatası kemiği mi olduğunu sorgulamamaktadır. Bu durum iki farklı senaryonun aynı kararı almasına yol açar:

- **Senaryo A:** Az beyin dokusu var → `t_white=249` doğru, dokuyu korur ✅
- **Senaryo B:** Az ama parlak kafatası var → `t_white=249` yanlış, 234–248 arası kafatası pikselleri silinmez ❌

Dahası, `MAX_STRIP_DEPTH` sorunu çözülse bile bu mekanizma varlığını sürdürdüğü sürece Senaryo B'deki slice'larda kafatası artıkları korunmaya devam edecektir. V1'de ince/normal dilim ayrımı yapılmadığından tüm dilimlere `t_white = max(200, 0.90 × max)` uygulanmış; bu slice'lardaki kafatası pikselleri temizlenmiştir.

---

## 10. V1 ile Karşılaştırma

| Özellik | V1 | V2 |
|---|---|---|
| Derinlik sınırı | Yok | 25 piksel |
| `t_white` formülü | Sabit `max(200, 0.90×mx)` | Adaptif: ince 249, normal 234 |
| Yön etkileşimi | Tek paylaşımlı `out` | 4 bağımsız maske |
| Okunan kaynak | `out` (mutate edilmiş) | `img_u8` (orijinal, sabit) |
| İstatistik alanı sayısı | 13 | 22 |
| Kalın halolu orta dilim | ✅ Temiz | ✅ Temiz |
| İnce/uç dilim (0049 tipi) | ✅ Tamamen temiz | ❌ Beyaz artık |

---

## 11. Sonuç

V2, V1'in paylaşımlı `out` dizisi sorununu bağımsız yön maskeleri ile başarıyla çözmüş ve istatistik zenginliğini önemli ölçüde artırmıştır. Ancak iki mekanizma pratikte beklenen faydayı sağlayamamıştır.

`MAX_STRIP_DEPTH=25` sınırı, kalın halosu olan slice'larda temizlemeyi yarıda kesmektedir. Bu sınır V1'in temel avantajını — derinlik sınırı olmaksızın halonun tamamını geçebilme — ortadan kaldırmaktadır.

`is_thin_slice` mekanizması ise non-background piksellerin içeriğini sorgulamadan yalnızca orana bakarak karar vermektedir. Sonuç olarak kafatası piksellerini beyin dokusu gibi "koruması" gereken durumlarda bile silinmeden bırakmaktadır.

Bu iki sorun birbirinden bağımsızdır: `MAX_STRIP_DEPTH` kaldırılsa bile `is_thin_slice` mekanizması 249 altındaki kafatası piksellerini silmeye devam etmeyecektir. Dataset üzerindeki görsel karşılaştırma, V1'in bu iki mekanizma olmaksızın daha temiz sonuç ürettiğini ortaya koymuştur. V2, V1'e kıyasla bir gerileme olarak değerlendirilmektedir.
