from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import csv

import pandas as pd          # ✅ BUNU EKLE
import joblib                # ✅ BUNU DA EKLE

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# -------------------------------------------------
#  Pydantic Modelleri
# -------------------------------------------------

class RiskRequest(BaseModel):
    il: str
    ilce: Optional[str] = ""
    yapimYili: int
    katSayisi: int
    zeminDukkan: str          # "evet" / "hayir"
    bitisik: str              # "evet" / "hayir"
    hasar: str                # "yok" / "hafif" / "kolon"

    kullanimAmaci: str        # "konut", "okul", "hastane", "sanayi", ...
    kisaKolon: str            # "yok" / "var" / "emin_degil"
    agirCikma: str            # "yok" / "hafif" / "buyuk"
    planTipi: str             # "dikdortgen" / "L" / "T" / "U" / "kompleks"
    bitisikHiza: Optional[str] = "yok"   # "uyumlu" / "farkli" / "yok"

    # Frontend'in hesapladığı AI çatlak skoru (şimdilik yoksa None)
    crackPuan: Optional[int] = None

    # Şimdilik kullanıcıdan istemiyoruz, backend tahmin edecek
    zeminSinifi: Optional[str] = None


class DepremInfo(BaseModel):
    seviye: str  # "düşük" / "orta" / "yüksek"
    puan: int


class YapisalInfo(BaseModel):
    seviye: str  # "Düşük" / "Orta" / "Yüksek"
    puan: int
    detaylar: List[str]


class RiskResponse(BaseModel):
    toplamPuan: int
    genelSeviye: str        # "Düşük" / "Orta" / "Yüksek"
    aciklama: str
    depremSeviye: str
    depremPuan: int
    yapisalSeviye: str
    yapisalPuan: int
    zeminSinifi: Optional[str]
    detaylar: List[str]
    # 🔹 Yeni alan: (AI’nin tahmin ettiği global risk etiketi)
    aiEtiket: Optional[str] = None
    aiYorum: Optional[str] = None


# -------------------------------------------------
#  UYGULAMA ve CORS
# -------------------------------------------------

app = FastAPI(title="Yapı Sağlığı Ön Tarama API")

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
#  DEPREM TEHLİKESİ HARİTASI
# -------------------------------------------------

DEPREM_RISK_MAP = {
    # --- MARMARA BÖLGESİ ---
    "istanbul": {
        "_default": "yüksek",  # Güney kıyıları ve fay hattına yakınlık
        "arnavutköy": "düşük", # Kuzeye gittikçe zemin ve uzaklık avantajı
        "sarıyer": "orta",
        "eyüpsultan": "orta", # Kuzey kısımları
        "beykoz": "düşük",
        "şile": "düşük",
        "çatalca": "orta",
        "başakşehir": "orta",
        "avcılar": "yüksek",
        "bakırköy": "yüksek",
        "küçükçekmece": "yüksek",
        "zeytinburnu": "yüksek",
        "fatih": "yüksek",
        "kadıköy": "yüksek",
        "maltepe": "yüksek",
        "kartal": "yüksek",
        "tuzla": "yüksek",
        "pendik": "yüksek",
        "adalar": "yüksek",
        "büyükçekmece": "yüksek",
        "silivri": "yüksek",
        "esenler": "orta",
        "sultangazi": "orta"
    },
    "kocaeli": {
        "_default": "yüksek", # KAF (Kuzey Anadolu Fayı) hattı üzerinde
        "kandıra": "orta",    # Karadeniz kıyısı daha güvenli
    },
    "sakarya": {
        "_default": "yüksek", # Tamamen fay hattı üzeri
    },
    "yalova": {
        "_default": "yüksek",
    },
    "bursa": {
        "_default": "yüksek",
        "büyükorhan": "orta",
        "harmancık": "orta",
        "keles": "orta",
    },
    "tekirdağ": {
        "_default": "yüksek", # Marmara fayı etkisi
        "saray": "düşük",
        "kapaklı": "orta",
    },
    "balıkesir": {
        "_default": "yüksek",
        "kepsut": "orta",
        "dursunbey": "orta",
    },
    "çanakkale": {
        "_default": "yüksek", # Biga yarımadası fayları
        "bozcaada": "orta",
    },
    "edirne": {
        "_default": "düşük", # Deprem riski az olan nadir illerden
        "enez": "orta",      # Saros körfezi fayı yakınlığı
        "ipsala": "orta",
    },
    "kırklareli": {
        "_default": "düşük", # Masif arazi, risk çok düşük
    },

    # --- EGE BÖLGESİ (BAF Sistemi - Çok Kırıklı Yapı) ---
    "izmir": {
        "_default": "yüksek",
        "kiraz": "orta",
        "beydağ": "orta",
        "karaburun": "orta", # Görece daha kayalık zemin
    },
    "manisa": {
        "_default": "yüksek",
        "demirci": "orta",
        "gördes": "orta",
    },
    "aydın": {
        "_default": "yüksek", # Graben sistemi
    },
    "muğla": {
        "_default": "yüksek",
        "datça": "orta", # Tarihsel aktivite yüksek olsa da zemin yapısı farklılaşabiliyor
    },
    "denizli": {
        "_default": "yüksek",
    },
    "uşak": {
        "_default": "orta",
    },
    "kütahya": {
        "_default": "yüksek", # Simav fayı vb.
    },
    "afyonkarahisar": {
        "_default": "orta",
        "dinar": "yüksek", # Tarihsel büyük deprem bölgesi
        "çay": "yüksek",
        "sultandağı": "yüksek",
    },

    # --- İÇ ANADOLU BÖLGESİ ---
    "ankara": {
        "_default": "düşük",
        "elmadağ": "orta",
        "nallıhan": "yüksek", # KAF'a yakın
        "beypazarı": "orta",
        "kazan": "orta",
        "kızılcahamam": "yüksek", # KAF'a yakın
        "çamlıdere": "yüksek",
        "evren": "orta",
        "bala": "orta",
    },
    "eskişehir": {
        "_default": "orta", # Şehir merkezinden fay geçiyor
        "inönü": "yüksek",
    },
    "konya": {
        "_default": "düşük", # Türkiye'nin en güvenli bölgelerinden
        "akşehir": "yüksek", # Fay zonu üzerinde
        "tuzlukçu": "yüksek",
        "ılgın": "orta",
        "doğanhisar": "orta",
    },
    "kayseri": {
        "_default": "orta", # Ecemiş fayı etkisi
        "yahyalı": "orta",
        "sarız": "orta", # DAF'a yaklaşım
    },
    "sivas": {
        "_default": "orta",
        "suşehri": "yüksek",    # KAF hattı (Kelkit vadisi)
        "koyulhisar": "yüksek", # KAF hattı
        "gölova": "yüksek",
        "akıncılar": "yüksek",
        "divriği": "düşük",
    },
    "aksaray": {
        "_default": "düşük", # Tuz gölü fayı tartışmalı ama genelde düşük/orta kabul edilir
    },
    "karaman": {
        "_default": "düşük", # En güvenli illerden biri
    },
    "niğde": {
        "_default": "orta",
        "bor": "orta",
    },
    "çankırı": {
        "_default": "yüksek", # KAF hattı üzerinde
        "yapraklı": "orta",
    },
    "kırşehir": {
        "_default": "orta",
    },
    "yozgat": {
        "_default": "orta",
        "akdağmadeni": "düşük",
    },

    # --- AKDENİZ BÖLGESİ ---
    "antalya": {
        "_default": "orta",
        "demre": "yüksek", # Fethiye-Burdur fay zonu etkisi
        "finike": "yüksek",
        "kumluca": "yüksek",
        "kemer": "yüksek",
        "kaş": "yüksek",
        "alanya": "düşük", # Masif bölgeye yakın
        "gündoğmuş": "düşük",
        "gazipaşa": "düşük"
    },
    "adana": {
        "_default": "yüksek", # Ceyhan, Yumurtalık, Karataş riskli. Zemin sıvılaşması riski yüksek.
        "tufanbeyli": "orta",
        "saimbeyli": "orta",
        "pozantı": "orta", # Dağlık kesim zemin avantajı olabilir ama fay hatlarına yakın
    },
    "mersin": {
        "_default": "düşük", # Mersin merkezi güvenli kabul edilir
        "tarsus": "orta",    # Adana havzasına ve Ecemiş fayına yakınlık
        "çamlıyayla": "orta",
    },
    "hatay": {
        "_default": "yüksek", # DAF ve Ölü Deniz Fayı kesişimi (Çok Yüksek Risk)
    },
    "kahramanmaraş": {
        "_default": "yüksek", # DAF Üzeri (Çok Yüksek Risk)
        "ekinözü": "orta",
    },
    "osmaniye": {
        "_default": "yüksek",
    },
    "burdur": {
        "_default": "yüksek", # Burdur Fayı
    },
    "isparta": {
        "_default": "yüksek",
    },

    # --- KARADENİZ BÖLGESİ ---
    "bolu": {
        "_default": "yüksek", # KAF hattı
        "seben": "orta",
        "kıbrıscık": "orta",
    },
    "düzce": {
        "_default": "yüksek", # KAF hattı
    },
    "zonguldak": {
        "_default": "düşük", # Kıyı şeridi nispeten güvenli
        "devrek": "yüksek",  # İç kesim faya yakın
        "gökçebey": "yüksek",
    },
    "bartın": {
        "_default": "düşük", # Amasra vb. güvenli
        "ulus": "orta",
    },
    "kastamonu": {
        "_default": "orta",
        "tosya": "yüksek", # KAF hattı tam üzerinden geçer
        "cide": "düşük",
    },
    "sinop": {
        "_default": "düşük", # Türkiye'nin en güvenli bölgelerinden (Kıyı şeridi)
        "boyabat": "yüksek", # KAF hattına yakın iç kesim
        "durağan": "yüksek",
    },
    "samsun": {
        "_default": "orta", # Merkez ve kıyı orta/düşük
        "ladik": "yüksek",  # KAF hattı tam üzerinden geçer
        "havza": "yüksek",
        "vezirköprü": "yüksek",
        "kavak": "yüksek",
        "bafra": "düşük", # Kıyı
        "alaçam": "düşük",
    },
    "ordu": {
        "_default": "düşük", # Kıyı şeridi
        "mesudiye": "yüksek", # KAF hattına yakın (Reşadiye sınırı)
        "akkuş": "yüksek",
    },
    "giresun": {
        "_default": "düşük",
        "alucra": "yüksek", # Şebinkarahisar fayı
        "şebinkarahisar": "yüksek", # KAF hattı
        "çamoluk": "yüksek",
    },
    "trabzon": {
        "_default": "düşük",
        "of": "orta",
        "hayrat": "orta",
    },
    "rize": {
        "_default": "düşük",
    },
    "artvin": {
        "_default": "düşük", # Kıyı
        "yusufeli": "orta",
    },
    "tokat": {
        "_default": "yüksek", # KAF hattı (Niksar, Erbaa, Reşadiye)
        "artova": "orta",
    },
    "amasya": {
        "_default": "yüksek", # KAF'a çok yakın ve tali faylar var
    },
    "çorum": {
        "_default": "orta",
        "osmancık": "yüksek", # KAF hattı
        "kargı": "yüksek",
        "mecitözü": "yüksek",
    },

    # --- DOĞU VE GÜNEYDOĞU ANADOLU (DAF ve KAF Birleşimi) ---
    "erzincan": {
        "_default": "yüksek", # KAF (Çok Yüksek Risk)
    },
    "bingöl": {
        "_default": "yüksek", # KAF ve DAF kesişim noktası (Karlıova) - En riskli yerlerden
    },
    "tunceli": {
        "_default": "yüksek", # Fay hatları arasında sıkışmış bölge
        "pülümür": "yüksek",  # KAF'a çok yakın
    },
    "elazığ": {
        "_default": "yüksek", # DAF (Sivrice-Pütürge hattı)
    },
    "malatya": {
        "_default": "yüksek", # DAF
    },
    "adıyaman": {
        "_default": "yüksek", # DAF
    },
    "diyarbakır": {
        "_default": "orta", # İl merkezi nispeten uzak
        "lice": "yüksek",   # Bitlis-Zagros bindirme kuşağı
        "hani": "yüksek",
        "çermik": "yüksek",
    },
    "şanlıurfa": {
        "_default": "düşük", # Arabistan levhası masif yapısı
        "bozova": "orta",    # Adıyaman sınırı
    },
    "mardin": {
        "_default": "düşük",
    },
    "batman": {
        "_default": "orta",
        "sason": "yüksek",
    },
    "siirt": {
        "_default": "yüksek", # Bindirme kuşağı
    },
    "hakkari": {
        "_default": "yüksek", # Yüksekova-Şemdinli fayı
    },
    "van": {
        "_default": "yüksek", # Aktif faylar çok fazla
    },
    "muş": {
        "_default": "yüksek", # Varto fayı vb.
    },
    "bitlis": {
        "_default": "yüksek",
    },
    "ağrı": {
        "_default": "orta",
        "patnos": "yüksek",
        "doğubayazıt": "yüksek", # Tendürek fayı
    },
    "erzurum": {
        "_default": "yüksek", # Aşkale, Çat vb.
        "ispir": "orta",
    },
    "kars": {
        "_default": "orta",
        "kağızman": "yüksek",
    },
    "ığdır": {
        "_default": "yüksek", # Tuzluca fayı
    },
    "ardahan": {
        "_default": "orta",
        "göle": "yüksek",
    }

}


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def deprem_seviyesi_bul(il: str, ilce: str) -> str:
    il_key = _norm(il)
    ilce_key = _norm(ilce)
    il_data = DEPREM_RISK_MAP.get(il_key)
    if not il_data:
        return "orta"
    return il_data.get(ilce_key, il_data.get("_default", "orta"))


def deprem_seviyesi_puan(seviye: str) -> int:
    if seviye == "yüksek":
        return 3
    if seviye == "orta":
        return 1
    return 0  # düşük


# -------------------------------------------------
#  BASİT ZEMİN SINIFI TAHMİNİ
#  (Tamamen kaba, ileride AFAD verisiyle değiştirirsin)
# -------------------------------------------------

ZEMIN_SINIF_MAP = {
    # --- MARMARA BÖLGESİ ---
    "istanbul": {
        "_default": "Z3",       # İstanbul geneli karmaşık ama yerleşim yoğunluğu orta zeminde
        "arnavutköy": "Z2",     # Kuzey bölgeler daha kayalık
        "sarıyer": "Z1",        # Zekeriyaköy/Masif araziler
        "eyüpsultan": "Z2",     # Kemerburgaz tarafı sağlam
        "beykoz": "Z1",         # Anadolu yakası kuzeyi (Kaya)
        "şile": "Z2",
        "çatalca": "Z2",
        "başakşehir": "Z2",     # Kayabaşı mevkii kayalık (Z1/Z2)
        "avcılar": "Z4",        # Kötü şöhretli, killi ve heyelan riski (Gürpınar vs)
        "bakırköy": "Z4",       # Kıyı şeridi zayıf
        "küçükçekmece": "Z4",   # Göl kenarı ve alüvyon
        "zeytinburnu": "Z3",
        "fatih": "Z2",          # Tarihi yarımada (kısmen kaya, kısmen dolgu)
        "kadıköy": "Z2",        # Kurbağalıdere hariç geneli sağlam
        "maltepe": "Z2",        # E-5 üstü sağlam, sahil dolgu
        "kartal": "Z1",         # Aydos tarafı kaya
        "tuzla": "Z3",
        "pendik": "Z2",
        "adalar": "Z1",         # Geneli kayalık
        "büyükçekmece": "Z4",   # Heyelan ve zayıf zemin
        "silivri": "Z3",
        "esenler": "Z2",
        "sultangazi": "Z2"
    },
    "kocaeli": {
        "_default": "Z3",
        "izmit": "Z4",          # Körfez ucu, alüvyon
        "kandıra": "Z2",
        "gebze": "Z2",          # Daha sert zemin
    },
    "sakarya": {
        "_default": "Z4",       # Adapazarı ovası (Sıvılaşma riski çok yüksek)
        "serdivan": "Z3",       # Tepelik kısımlar daha iyi
    },
    "yalova": {
        "_default": "Z4",       # Kıyı kesimi alüvyon
        "çınarcık": "Z3",
    },
    "bursa": {
        "_default": "Z3",
        "osmangazi": "Z4",      # Ova kısmı alüvyon
        "nilüfer": "Z3",        # Alüvyon/Kil karışık
        "yıldırım": "Z2",       # Uludağ etekleri sağlam
        "büyükorhan": "Z2",
        "harmancık": "Z2",
        "keles": "Z2",
        "gemlik": "Z4",         # Alüvyon, deniz kili
    },
    "tekirdağ": {
        "_default": "Z3",
        "çorlu": "Z2",
        "marmaraereğlisi": "Z3",
    },
    "balıkesir": {
        "_default": "Z3",
        "edremit": "Z4",        # Kıyı ovası
    },
    "çanakkale": {
        "_default": "Z3",
        "bozcaada": "Z2",
    },
    "edirne": {
        "_default": "Z3",       # Meriç havzası yumuşak
    },
    "kırklareli": {
        "_default": "Z1",       # Istranca masifi (Kaya)
    },

    # --- EGE BÖLGESİ ---
    "izmir": {
        "_default": "Z3",
        "bayraklı": "Z4",       # Alüvyon, delta (En riskli zeminlerden)
        "bornova": "Z4",        # Ova kısmı Z4, tepeler Z2
        "buca": "Z2",           # Kaynaklar tarafı sağlam
        "konak": "Z3",          # Kıyı dolgu Z4, varyant Z2
        "karşıyaka": "Z4",      # Bostanlı/Mavişehir (Alüvyon/Dolgu)
        "balçova": "Z4",        # Ilıca, alüvyon
        "kiraz": "Z2",
        "beydağ": "Z2",
        "karaburun": "Z1",      # Kayalık
    },
    "manisa": {
        "_default": "Z3",       # Gediz ovası
        "demirci": "Z2",
    },
    "aydın": {
        "_default": "Z4",       # Büyük Menderes Grabeni (Tamamen alüvyon)
        "kuşadası": "Z2",       # Kısmen kayalık
    },
    "muğla": {
        "_default": "Z2",       # Menteşe yöresi dağlık
        "dalaman": "Z4",        # Ova
        "datça": "Z1",          # Sert zemin
    },
    "denizli": {
        "_default": "Z3",
    },
    "kütahya": {
        "_default": "Z2",
    },
    "afyonkarahisar": {
        "_default": "Z3",
        "dinar": "Z4",          # Alüvyon
    },

    # --- İÇ ANADOLU BÖLGESİ ---
    "ankara": {
        "_default": "Z2",       # Ankara geneli step ve sert zemin
        "çankaya": "Z1",        # Oran, Dikmen tarafları kaya
        "etimesgut": "Z4",      # Ankara çayı havzası (Gevşek zemin)
        "sincan": "Z3",
        "yenimahalle": "Z3",    # Demetevler (Daha zayıf), Batıkent (Orta)
        "mamak": "Z2",
        "keçiören": "Z2",       # Geneli sağlam
        "gölbaşı": "Z4",        # Göl çevresi, bataklık zemini
        "elmadağ": "Z2",
        "kızılcahamam": "Z2",
    },
    "eskişehir": {
        "_default": "Z4",       # Porsuk çayı ovası (Merkez alüvyon)
        "odunpazarı": "Z2",     # Yamaçlar sağlam
    },
    "konya": {
        "_default": "Z4",       # Eski göl tabanı (Alüvyon)
        "meram": "Z3",
        "selçuklu": "Z3",
    },
    "kayseri": {
        "_default": "Z3",       # Erciyes etekleri volkanik tüf (Değişken)
        "melikgazi": "Z2",
    },
    "sivas": {
        "_default": "Z2",
    },
    "aksaray": {
        "_default": "Z3",
    },
    "karaman": {
        "_default": "Z2",
    },
    "niğde": {
        "_default": "Z3",
    },
    "çankırı": {
        "_default": "Z3",
    },
    "kırşehir": {
        "_default": "Z2",
    },
    "yozgat": {
        "_default": "Z2",
    },

    # --- AKDENİZ BÖLGESİ ---
    "antalya": {
        "_default": "Z1",       # Merkez traverten plato (Çok sağlam kaya)
        "konyaaltı": "Z3",      # Çakıl/Kum
        "lara": "Z1",           # Falezler (Kaya)
        "demre": "Z4",          # Delta ovası
        "finike": "Z4",
        "kumluca": "Z4",
        "alanya": "Z2",
        "manavgat": "Z3",
    },
    "adana": {
        "_default": "Z4",       # Çukurova Deltası (Derin alüvyon - Yüksek Risk)
        "çukurova": "Z2",       # Kuzey kesimler (Daha sağlam)
        "seyhan": "Z4",
        "yüreğir": "Z4",
    },
    "mersin": {
        "_default": "Z3",       # Kıyı şeridi dolgu/alüvyon
        "toroslar": "Z2",       # Dağ eteği
    },
    "hatay": {
        "_default": "Z4",       # Amik Ovası (Alüvyon - Çok yüksek sıvılaşma)
        "antakya": "Z4",
        "iskenderun": "Z4",     # Kıyı şeridi ve dolgu alanları
        "belen": "Z2",          # Dağlık geçit
    },
    "kahramanmaraş": {
        "_default": "Z3",       # Ovası Z4, yamaçlar Z2
    },
    "osmaniye": {
        "_default": "Z4",
    },
    "burdur": {
        "_default": "Z3",       # Göl kenarı gevşek
    },
    "isparta": {
        "_default": "Z3",
    },

    # --- KARADENİZ BÖLGESİ ---
    "bolu": {
        "_default": "Z3",
    },
    "düzce": {
        "_default": "Z4",       # Düzce Ovası (Alüvyon - Yüksek sıvılaşma)
    },
    "zonguldak": {
        "_default": "Z2",       # Eğimli ve kayalık arazi
    },
    "bartın": {
        "_default": "Z3",
    },
    "kastamonu": {
        "_default": "Z2",
    },
    "sinop": {
        "_default": "Z2",
    },
    "samsun": {
        "_default": "Z3",
        "atakum": "Z3",
        "bafra": "Z4",          # Kızılırmak deltası
        "çarşamba": "Z4",       # Yeşilırmak deltası
        "canik": "Z2",
    },
    "ordu": {
        "_default": "Z2",       # Yamaçlar genelde sağlam
    },
    "giresun": {
        "_default": "Z4",
    },
    "trabzon": {
        "_default": "Z2",       # Volkanik kayaçlar yaygın
        "ortahisar": "Z3",      # Kıyı ve dere ağızları
        "of": "Z3",
    },
    "rize": {
        "_default": "Z2",       # Dağlık, ancak heyelan riski var
    },
    "artvin": {
        "_default": "Z1",       # Çok sarp ve kayalık
    },
    "tokat": {
        "_default": "Z3",
        "erbaa": "Z4",          # Ova
        "niksar": "Z4",         # Ova
    },
    "amasya": {
        "_default": "Z3",       # Yeşilırmak kıyısı
    },
    "çorum": {
        "_default": "Z3",
    },

    # --- DOĞU VE GÜNEYDOĞU ANADOLU ---
    "erzincan": {
        "_default": "Z4",       # Erzincan Ovası (Derin alüvyon - Çok riskli)
    },
    "bingöl": {
        "_default": "Z3",       # Ova kısımları Z4
    },
    "tunceli": {
        "_default": "Z1",       # Sarp kayalık
    },
    "elazığ": {
        "_default": "Z2",       # Harput tarafı Z1, Merkez Z3
    },
    "malatya": {
        "_default": "Z3",       # Kayısı bahçeleri/ova Z4
        "yeşilyurt": "Z3",
    },
    "adıyaman": {
        "_default": "Z3",
    },
    "diyarbakır": {
        "_default": "Z2",       # Geniş bazalt platoları (Sağlam kaya)
        "bağlar": "Z2",
        "yenişehir": "Z3",      # Dicle kenarı alüvyon
    },
    "şanlıurfa": {
        "_default": "Z2",       # Kireçtaşı platoları
    },
    "mardin": {
        "_default": "Z1",       # Mardin Eşiği (Masif kaya)
    },
    "batman": {
        "_default": "Z3",
    },
    "siirt": {
        "_default": "Z2",
    },
    "hakkari": {
        "_default": "Z1",       # Çok dağlık
    },
    "van": {
        "_default": "Z3",       # Göl kıyıları Z4
        "erciş": "Z4",
    },
    "muş": {
        "_default": "Z4",       # Muş Ovası
    },
    "bitlis": {
        "_default": "Z1",       # Volkanik kayaçlar
        "tatvan": "Z3",
    },
    "ağrı": {
        "_default": "Z3",
        "doğubayazıt": "Z4",    # Ova
    },
    "erzurum": {
        "_default": "Z3",       # Ovası Z4, Palandöken Z2
        "yakutiye": "Z3",
    },
    "kars": {
        "_default": "Z2",
    },
    "ığdır": {
        "_default": "Z4",       # Iğdır Ovası (Alüvyon)
    },
    "ardahan": {
        "_default": "Z2",
    }

}


def tahmini_zemin_sinifi(il: str, ilce: str) -> Optional[str]:
    il_key = _norm(il)
    ilce_key = _norm(ilce)
    il_data = ZEMIN_SINIF_MAP.get(il_key)
    if not il_data:
        return None
    return il_data.get(ilce_key, il_data.get("_default"))




def ai_yorum_uret(
    req: RiskRequest,
    zemin_sinifi: Optional[str],
    genel_seviye: str,
    ai_etiket: Optional[str],
) -> Optional[str]:
    """
    AI modelinin tahminiyle ilgili, mühendis gibi kısa bir yorum üretir.
    """
    if ai_etiket is None:
        return None

    nedenler: List[str] = []

    # 1) Yapım yılı
    if req.yapimYili <= 1999:
        nedenler.append("1999 öncesi inşa edilmiş olması")
    elif req.yapimYili <= 2018 and ai_etiket != "Düşük":
        nedenler.append("1999–2018 arası inşa edilmiş olması")

    # 2) Kat sayısı
    if req.katSayisi >= 8:
        nedenler.append("yüksek kat sayısı (8+ kat)")
    elif req.katSayisi >= 4 and ai_etiket != "Düşük":
        nedenler.append("orta katlı bir bina olması")

    # 3) Zemin katta dükkân
    if req.zeminDukkan == "evet":
        nedenler.append("zemin katta dükkân / geniş açıklık bulunması")

    # 4) Gözle hasar
    if req.hasar == "kolon":
        nedenler.append("taşıyıcı elemanlarda (kolon/kiriş) hasar bildirilmesi")
    elif req.hasar == "hafif":
        nedenler.append("gözle görülür çatlakların bulunması")

    # 5) Kısa kolon
    if req.kisaKolon == "var":
        nedenler.append("kısa kolon riskinin olması")

    # 6) Ağır çıkma
    if req.agirCikma == "buyuk":
        nedenler.append("büyük konsol/çıkma bulunması")
    elif req.agirCikma == "hafif" and ai_etiket != "Düşük":
        nedenler.append("hafif de olsa çıkma bulunması")

    # 7) Bitişik bina kat hizası
    if req.bitisikHiza == "farkli":
        nedenler.append("bitişik binalarla kat hizalarının farklı olması")

    # 8) Kullanım amacı
    if req.kullanimAmaci in ("okul", "hastane"):
        nedenler.append("kullanımın okul/hastane gibi kritik olması")

    # 9) Zemin sınıfı
    if zemin_sinifi in ("Z3", "Z4"):
        nedenler.append(f"zemin sınıfının {zemin_sinifi} (yumuşak/yumuşakça zemin) olması")
    elif zemin_sinifi == "Z2" and ai_etiket != "Düşük":
        nedenler.append("zemin sınıfının orta kalitede (Z2) olması")

    # Birkaç sebep seçelim ki çok uzamasın
    if not nedenler:
        neden_metin = "benzer binalardan öğrendiği istatistiksel örüntüler"
    else:
        neden_metin = ", ".join(nedenler[:3])

    # AI ile kural tabanlı skor aynı mı, farklı mı?
    if ai_etiket == genel_seviye:
        return (
            f"Yapay zekâ modeli de kural tabanlı sistem gibi bu binayı "
            f"'{ai_etiket}' risk sınıfına yerleştiriyor. Özellikle {neden_metin} "
            "gibi özellikler bu sınıflandırmayı destekliyor."
        )
    else:
        return (
            f"Kural tabanlı sistem bu binayı '{genel_seviye}' risk sınıfına koyarken, "
            f"yapay zekâ modeli '{ai_etiket}' tahmini yapıyor. Model, özellikle "
            f"{neden_metin} gibi özelliklere bakarak bu binayı benzerlerinden daha "
            f"'{ai_etiket.lower()}' risk grubuna yakın görüyor."
        )



def ai_risk_tahmin(req: RiskRequest, zemin_sinifi: Optional[str]) -> Optional[str]:
    """
    Eğitilmiş RandomForest modelini kullanarak
    Düşük / Orta / Yüksek tahmini yapar.
    """
    if ml_pipeline is None:
        return None  # model yüklenmemişse None döneriz

    df = pd.DataFrame([{
        "il": req.il,
        "ilce": req.ilce,
        "yapimYili": req.yapimYili,
        "katSayisi": req.katSayisi,
        "zeminDukkan": req.zeminDukkan,
        "bitisik": req.bitisik,
        "hasar": req.hasar,
        "kullanimAmaci": req.kullanimAmaci,
        "kisaKolon": req.kisaKolon,
        "agirCikma": req.agirCikma,
        "planTipi": req.planTipi,
        "bitisikHiza": req.bitisikHiza,
        "zeminSinifi": zemin_sinifi or "bilinmiyor",
    }])

    # Modelin predict'i şu an zaten "Düşük" / "Orta" / "Yüksek" stringi döndürüyor
    y_pred = ml_pipeline.predict(df)[0]
    return str(y_pred)


# -------------------------------------------------
#  YAPISAL SKOR HESABI (mühendislik ağırlıklı katsayılar)
# -------------------------------------------------

from typing import Tuple, List, Optional

def yapisal_skor_hesapla(req: RiskRequest, zemin_sinifi: Optional[str]) -> Tuple[int, List[str]]:
    """
    Yapısal risk için yaklaşık 0–40 arası bir ön skor üretir.
    Bu yalnızca ön tarama amacıyla kullanılır, resmî deprem performans
    analizinin yerini TUTMAZ.
    """
    skor = 0
    detaylar: List[str] = []

    # 1) Yapım yılı (dolaylı olarak yönetmelik seviyesi)
    # Türkiye'de 1975, 1998, 2007, 2018 yönetmelik değişimleri var.
    if req.yapimYili <= 1975:
        skor += 5
        detaylar.append(f"{req.yapimYili} ve öncesi – çok eski betonarme (+5)")
    elif req.yapimYili <= 1999:
        skor += 4
        detaylar.append(f"{req.yapimYili} – 1999 öncesi, eski deprem yönetmeliği (+4)")
    elif req.yapimYili <= 2018:
        skor += 2
        detaylar.append(f"{req.yapimYili} – 1999–2018 arası, orta seviye yönetmelik (+2)")
    else:
        skor += 1
        detaylar.append(f"{req.yapimYili} – 2018 sonrası, güncel yönetmeliğe yakın (+1)")

    # 2) Kat sayısı (yükseklik arttıkça risk artıyor)
    if req.katSayisi <= 3:
        detaylar.append(f"{req.katSayisi} kat – az katlı (+0)")
    elif req.katSayisi <= 5:
        skor += 1
        detaylar.append(f"{req.katSayisi} kat – orta yükseklik (+1)")
    elif req.katSayisi <= 8:
        skor += 2
        detaylar.append(f"{req.katSayisi} kat – yüksek yapı sınıfına yakın (+2)")
    else:
        skor += 4
        detaylar.append(f"{req.katSayisi} kat – yüksek yapı (+4)")

    # 3) Zemin katta dükkân / soft-story etkisi
    if req.zeminDukkan == "evet":
        skor += 3
        detaylar.append("Zemin katta dükkân / geniş açıklık – soft story riski (+3)")
    else:
        detaylar.append("Zemin katta dükkân yok (+0)")

    # 4) Bitişik nizam (çarpışma – pounding riski)
    if req.bitisik == "evet":
        skor += 1
        detaylar.append("Bitişik nizam – yan bina ile çarpışma riski (+1)")
    else:
        detaylar.append("Ayrık nizam (+0)")

    # 5) Mevcut hasar durumu (en kritik göstergelerden biri)
    if req.hasar == "hafif":
        skor += 4
        detaylar.append("Belirgin hafif çatlaklar var (+4)")
    elif req.hasar == "kolon":
        skor += 8
        detaylar.append("Kolonda / taşıyıcıda ciddi hasar var (+8)")
    else:
        detaylar.append("Belirgin hasar yok (+0)")

    # 6) Kullanım amacı (kritik binalara biraz daha hassas yaklaş)
    if req.kullanimAmaci in ("okul", "hastane"):
        skor += 2
        detaylar.append(f"Kritik kullanım ({req.kullanimAmaci}) – performans beklentisi yüksek (+2)")
    elif req.kullanimAmaci in ("sanayi", "ticari"):
        skor += 1
        detaylar.append(f"Sanayi / ticari kullanım ({req.kullanimAmaci}) (+1)")
    else:
        detaylar.append(f"Konut / normal kullanım ({req.kullanimAmaci}) (+0)")

    # 7) Kısa kolon durumu
    if req.kisaKolon == "var":
        skor += 3
        detaylar.append("Kısa kolon var – gevrek kırılma riski (+3)")
    elif req.kisaKolon == "emin_degil":
        skor += 1
        detaylar.append("Kısa kolon olma ihtimali var, belirsiz (+1)")
    else:
        detaylar.append("Kısa kolon yok (+0)")

    # 8) Ağır çıkma / balkon / teras
    if req.agirCikma == "buyuk":
        skor += 2
        detaylar.append("Belirgin ağır çıkma var (+2)")
    elif req.agirCikma == "hafif":
        skor += 1
        detaylar.append("Hafif çıkma var (+1)")
    else:
        detaylar.append("Belirgin ağır çıkma yok (+0)")

    # 9) Plan düzensizliği (L, T, U, kompleks şekiller)
    if req.planTipi == "dikdortgen":
        detaylar.append("Dikdörtgen / düzenli plan (+0)")
    elif req.planTipi in ("L", "T", "U"):
        skor += 2
        detaylar.append(f"{req.planTipi} tipi düzensiz plan (+2)")
    elif req.planTipi == "kompleks":
        skor += 3
        detaylar.append("Karmaşık / çok düzensiz plan (+3)")
    else:
        detaylar.append("Plan tipi bilinmiyor (+0)")

    # 10) Bitişik bina kat hizası (pounding etkisini büyütür)
    if req.bitisikHiza == "farkli":
        skor += 2
        detaylar.append("Yan bina ile kat hizası farklı – çarpışma riski (+2)")
    elif req.bitisikHiza == "uyumlu":
        detaylar.append("Yan bina ile kat hizası uyumlu (+0)")
    else:
        detaylar.append("Bitişik bina yok / bilgi yok (+0)")

    # 11) Çatlak analizi (görüntü işleme / AI'den gelen crackPuan: 0–3)
    cp = getattr(req, "crackPuan", None)
    if cp is not None:
        # 0 → 0, 1 → 2, 2 → 3, 3+ → 4
        cp_clamped = max(0, min(3, cp))
        crack_skor_map = {0: 0, 1: 2, 2: 3, 3: 4}
        crack_skor = crack_skor_map[cp_clamped]
        skor += crack_skor

        if cp_clamped == 0:
            detaylar.append("Çatlak analizi: belirgin risk görülmedi (+0)")
        elif cp_clamped == 1:
            detaylar.append(f"Çatlak analizi: hafif seviyede çatlaklar (+{crack_skor})")
        elif cp_clamped == 2:
            detaylar.append(f"Çatlak analizi: orta seviyede çatlaklar (+{crack_skor})")
        else:  # 3
            detaylar.append(f"Çatlak analizi: yaygın / ciddi çatlaklar (+{crack_skor})")
    else:
        detaylar.append("Çatlak analizi verisi yok (+0)")

    # 12) Zemin sınıfı (tahmini)
    if zemin_sinifi in ("Z3", "Z4"):
        skor += 2
        detaylar.append(f"Zemin sınıfı {zemin_sinifi} – yumuşak/zayıf zemin (+2)")
    elif zemin_sinifi == "Z2":
        skor += 1
        detaylar.append("Zemin sınıfı Z2 – orta zemin (+1)")
    elif zemin_sinifi == "Z1":
        detaylar.append("Zemin sınıfı Z1 – görece sağlam zemin (+0)")
    else:
        detaylar.append("Zemin sınıfı bilinmiyor (+0)")

    return skor, detaylar


def yapisal_seviye_etiketi(skor: int) -> str:
    if skor <= 3:
        return "Düşük"
    if skor <= 10:
        return "Orta"
    return "Yüksek"


def genel_risk_seviyesi(toplam_puan: int) -> str:
    if toplam_puan <= 7:
        return "Düşük"
    if toplam_puan <= 15:
        return "Orta"
    return "Yüksek"


def aciklama_uret(genel_seviye: str, deprem_seviye: str, yapisal_seviye: str) -> str:
    if genel_seviye == "Yüksek":
        return (
            f"Ön taramaya göre risk seviyesi yüksektir. "
            f"Bina {deprem_seviye} deprem tehlikesi bölgesinde ve "
            f"form verilerine göre yapısal risk {yapisal_seviye.lower()} seviyededir. "
            "Lisanslı mühendislik bürosuna başvurmanız önerilir."
        )
    elif genel_seviye == "Orta":
        return (
            "Ön taramaya göre risk seviyesi orta düzeydedir. "
            "Deprem güvenliği açısından detaylı mühendislik değerlendirmesi yapılması uygundur."
        )
    else:
        return (
            "Ön taramaya göre risk seviyesi düşüktür. "
            "Yine de bina güvenliği için yetkili mühendis tarafından kontrol edilmesi tavsiye edilir."
        )


# -------------------------------------------------
#  VERİ KAYDI (CSV) – AI için dataset
# -------------------------------------------------

DATA_FILE = Path("veri_kayitlari.csv")

HEADER = [
    "tarih",
    "il",
    "ilce",
    "yapimYili",
    "katSayisi",
    "zeminDukkan",
    "bitisik",
    "hasar",
    "kullanimAmaci",
    "kisaKolon",
    "agirCikma",
    "planTipi",
    "bitisikHiza",
    "zeminSinifi",
    "crackPuan",
    "depremSeviye",
    "depremPuan",
    "yapisalPuan",
    "yapisalSeviye",
    "genelSeviye",
    "gercekEtiket",
]


def kayit_ekle_csv(
    req: RiskRequest,
    zemin_sinifi: Optional[str],
    deprem_seviye: str,
    deprem_puan: int,
    yapisal_puan: int,
    yapisal_seviye: str,
    genel_seviye: str,
):
    yeni_dosya = not DATA_FILE.exists()

    with DATA_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if yeni_dosya:
            writer.writerow(HEADER)

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            req.il,
            req.ilce,
            req.yapimYili,
            req.katSayisi,
            req.zeminDukkan,
            req.bitisik,
            req.hasar,
            req.kullanimAmaci,
            req.kisaKolon,
            req.agirCikma,
            req.planTipi,
            req.bitisikHiza,
            zemin_sinifi or "",
            req.crackPuan if req.crackPuan is not None else 0,
            deprem_seviye,
            deprem_puan,
            yapisal_puan,
            yapisal_seviye,
            genel_seviye,
            "",  # gercekEtiket – mühendis sonradan dolduracak
        ])

# -------------------------------------------------
#  EĞİTİLMİŞ MAKİNE ÖĞRENMESİ MODELİ
# -------------------------------------------------

try:
    ml_pipeline = joblib.load("risk_model.joblib")
    label_encoder = joblib.load("etiket_encoder.joblib")
    print("ML modeli ve etiket encoder yüklendi.")
except Exception as e:
    print("ML modeli yüklenemedi, kural bazlı sistem devam edecek:", e)
    ml_pipeline = None
    label_encoder = None

# -------------------------------------------------
#  ENDPOINTLER
# -------------------------------------------------

@app.get("/")
def root():
    return {"message": "Yapı Sağlığı Ön Tarama API çalışıyor."}


@app.post("/api/risk-hesapla", response_model=RiskResponse)
def risk_hesapla(req: RiskRequest):
    # Deprem tehlikesi
    deprem_seviye = deprem_seviyesi_bul(req.il, req.ilce)
    deprem_puan = deprem_seviyesi_puan(deprem_seviye)

    # Zemin sınıfı tahmini
    zemin_sinifi = tahmini_zemin_sinifi(req.il, req.ilce)

    # Yapısal skor
    yapisal_puan, detaylar = yapisal_skor_hesapla(req, zemin_sinifi)
    yapisal_seviye = yapisal_seviye_etiketi(yapisal_puan)

    # Genel skor
    toplam = deprem_puan + yapisal_puan
    genel_seviye = genel_risk_seviyesi(toplam)
    aciklama = aciklama_uret(genel_seviye, deprem_seviye, yapisal_seviye)
    # 🔹 AI modelinden global risk etiketi al
    ai_etiket = ai_risk_tahmin(req, zemin_sinifi)
    ai_yorum = ai_yorum_uret(
        req=req,
        zemin_sinifi=zemin_sinifi,
        genel_seviye=genel_seviye,
        ai_etiket=ai_etiket,
    )
    # CSV'ye kaydet – AI verisi
    kayit_ekle_csv(
        req=req,
        zemin_sinifi=zemin_sinifi,
        deprem_seviye=deprem_seviye,
        deprem_puan=deprem_puan,
        yapisal_puan=yapisal_puan,
        yapisal_seviye=yapisal_seviye,
        genel_seviye=genel_seviye,
    )



    # Frontend'in kullandığı alan isimleriyle döndür
    return RiskResponse(
        toplamPuan=toplam,
        genelSeviye=genel_seviye,
        aciklama=aciklama,
        depremSeviye=deprem_seviye,
        depremPuan=deprem_puan,
        yapisalSeviye=yapisal_seviye,
        yapisalPuan=yapisal_puan,
        zeminSinifi=zemin_sinifi,
        detaylar=detaylar,
        aiEtiket=ai_etiket,
        aiYorum=ai_yorum,
    )
