# Yapı Sağlığı — Deprem Riski Ön Tarama Aracı

Bir binanın **deprem riskini ön taramadan geçiren** araç: kullanıcı binanın konumunu ve
yapısal özelliklerini bir formla girer, sistem deprem tehlikesi ile yapısal zafiyeti
ayrı ayrı puanlar ve **Düşük / Orta / Yüksek** seviyesinde bir sonuç ile gerekçeli bir
yorum döner.

FastAPI tabanlı bir servis + tek dosyalık bir web arayüzünden oluşur.

> ⚠️ **Bu bir ön tarama aracıdır, mühendislik raporu değildir.** Sonuçlar yapı
> denetimi, performans analizi veya resmî risk değerlendirmesi yerine geçmez. Gerçek
> bir değerlendirme için inşaat mühendisine başvurulmalıdır.

---

## Nasıl çalışır

Risk iki ayrı eksende puanlanır, sonra birleştirilir:

**1 · Deprem tehlikesi** — il ve ilçe, kodun içindeki tehlike haritasından
(`DEPREM_RISK_MAP`) bir seviyeye oturtulur. İlçe eşleşmesi bulunamazsa il seviyesine
düşülür. Konumdan ayrıca bir **zemin sınıfı tahmini** (`Z1`–`Z4`) üretilir.

**2 · Yapısal zafiyet** — formdan gelen özellikler ağırlıklı bir skora çevrilir
(`yapisal_skor_hesapla`) ve her ağırlık, skoru *neden* yükselttiğini söyleyen bir
gerekçe satırı bırakır:

| Girdi | Neden sorulur |
|-------|---------------|
| Yapım yılı | Deprem yönetmeliği kuşağı (1975 / 1998 / 2007 / 2018) |
| Kat sayısı | Yükseklik arttıkça talep artar |
| Zemin katta dükkân | **Yumuşak kat** — en yaygın göçme mekanizmalarından biri |
| Kısa kolon | Kirişsiz/bölünmüş kolon, kesme kırılması riski |
| Ağır çıkma | Kütlenin düşey taşıyıcı dışına taşması, burulma |
| Plan tipi | `L` / `T` / `U` / kompleks planlarda burulma düzensizliği |
| Bitişik nizam + kat hizası | Hizasız döşeme → **çarpışma (pounding)** etkisi |
| Gözle görülür hasar | Mevcut çatlak / kolon hasarı |
| Kullanım amacı | Okul, hastane gibi binalarda sonuç daha muhafazakâr |

İkisinin toplamı `genel_risk_seviyesi()` ile seviyeye çevrilir, `aciklama_uret()` ile
de insanın okuyabileceği bir yorum yazılır. Yani sonuç bir kara kutu değil: hangi
girdinin puanı yükselttiği gerekçe listesinde görünür.

### Makine öğrenmesi katmanı

Kural motorunun yanında eğitilmiş bir **RandomForest** sınıflandırıcı
(`risk_model.joblib`) ikinci bir görüş olarak çalışır. Model yüklenemezse sistem
**kural bazlı olarak çalışmaya devam eder** — ML bir bağımlılık değil, bir eklenti.

> **Dürüst sınır:** model, kural motorunun kendisinin ürettiği 5000 satırlık sentetik
> veriyle eğitildi (`sentetik_veri_uret.py` → `sentetik_bina_verisi.csv`). Yani
> öğrendiği şey gerçek saha sonuçları değil, **kuralların kendisidir**; yüksek test
> doğruluğu da bunu ölçer. Katmanın gerçek değeri gerçek etiketli saha verisiyle
> yeniden eğitildiğinde ortaya çıkar — `gercekEtiket` sütunu tam bunun için duruyor.

---

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt

# Servisi başlat (http://127.0.0.1:8000)
uvicorn main:app --reload --port 8000
```

Sonra `arayuz.html` dosyasını tarayıcıda aç — arayüz `127.0.0.1:8000` adresindeki
servise istek atar (CORS açık). Servisin tek başına çalıştığını görmek için:

```bash
curl http://127.0.0.1:8000/
# {"message":"Yapı Sağlığı Ön Tarama API çalışıyor."}
```

### Modeli yeniden eğitmek

```bash
python sentetik_veri_uret.py    # 5000 satirlik sentetik_bina_verisi.csv uretir
python train_model.py           # risk_model.joblib olarak kaydeder
```

`sentetik_veri_uret.py`, `main.py` içindeki **aynı** puanlama fonksiyonlarını
çağırarak veri üretir — yani etiketler kural motoruyla tanım gereği tutarlıdır.

---

## API

### `POST /api/risk-hesapla`

```jsonc
{
  "il": "istanbul",
  "ilce": "avcılar",
  "yapimYili": 1992,
  "katSayisi": 6,
  "zeminDukkan": "evet",        // evet | hayir
  "bitisik": "evet",            // evet | hayir
  "hasar": "hafif",             // yok | hafif | kolon
  "kullanimAmaci": "konut",     // konut | ofis | okul | hastane | sanayi
  "kisaKolon": "var",           // yok | var | emin_degil
  "agirCikma": "buyuk",         // yok | hafif | buyuk
  "planTipi": "L",              // dikdortgen | L | T | U | kompleks
  "bitisikHiza": "farkli",      // yok | uyumlu | farkli
  "crackPuan": 0                // opsiyonel: arayuzun catlak skoru
}
```

Yanıt; deprem bilgisi, yapısal puan ve **gerekçe listesi**, genel seviye, açıklama
metni ve (model yüklüyse) ML tahminini birlikte döner.

Her istek `veri_kayitlari.csv`'ye bir satır olarak eklenir (dosya yoksa başlığıyla
birlikte oluşturulur). Dosya bir çalışma çıktısıdır, depoda tutulmaz.

---

## Dosyalar

| Dosya | Görevi |
|-------|--------|
| `main.py` | FastAPI servisi: tehlike haritası, zemin tahmini, puanlama, yorum, kayıt |
| `arayuz.html` | Tek dosyalık web arayüzü (form + sonuç ekranı), bağımlılık yok |
| `sentetik_veri_uret.py` | Kural motorundan 5000 satırlık eğitim verisi üretir |
| `train_model.py` | RandomForest pipeline'ını eğitir ve `risk_model.joblib` yazar |
| `sentetik_bina_verisi.csv` | Üretilmiş eğitim verisi |
| `risk_model.joblib` · `etiket_encoder.joblib` | Eğitilmiş model ve etiket kodlayıcı |

---

## Bilinen sınırlar

- Tehlike haritası ve zemin sınıfı tahmini **il/ilçe çözünürlüğündedir**; bina
  parseli düzeyinde bir zemin etüdü yerine geçmez.
- ML katmanı sentetik veriyle eğitildi (yukarıdaki nota bakın).
- `veri_kayitlari.csv` bir dosyaya eklemeli yazılır; eşzamanlı yük için veritabanı gerekir.
- `risk_model.joblib` bir **pickle** dosyasıdır. Depo herkese açık olduğu için: pickle
  yüklemek, yükleyen makinede kod çalıştırmakla aynı şeydir — güvenmediğiniz bir
  kaynaktan gelen `.joblib` dosyasını açmayın. Bu depodaki modeli `train_model.py`
  ile kendiniz yeniden üretebilirsiniz.

## Lisans

Eğitim ve portföy amaçlı yayımlanmıştır; okuyabilir ve inceleyebilirsiniz.
Tüm hakları saklıdır — bkz. [`LICENSE`](LICENSE).
