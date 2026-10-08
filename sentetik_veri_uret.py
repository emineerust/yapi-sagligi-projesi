
import pandas as pd
import joblib
from pathlib import Path
import csv
import random

MODEL_PATH = Path("risk_model.joblib")
ENCODER_PATH = Path("etiket_encoder.joblib")

try:
    ml_pipeline = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except Exception:
    ml_pipeline = None
    label_encoder = None

# Backend'deki sınıf ve fonksiyonları kullanıyoruz
from main import (
    RiskRequest,
    yapisal_skor_hesapla,
    deprem_seviyesi_bul,
    deprem_seviyesi_puan,
    genel_risk_seviyesi,
)

# Kaç tane sentetik kayıt üretilecek
KAYIT_SAYISI = 5000

# Çıkacak dosya
OUTPUT_FILE = Path("sentetik_bina_verisi.csv")

ILLER_VE_ILCELER = {
    "istanbul": ["arnavutköy", "sarıyer", "eyüpsultan", "beykoz", "şile", "çatalca", "başakşehir", "avcılar", "bakırköy", "küçükçekmece", "zeytinburnu", "fatih", "kadıköy", "maltepe", "kartal", "tuzla", "pendik", "adalar", "büyükçekmece", "silivri", "esenler", "sultangazi"],
    "kocaeli": ["kandıra"],
    "bursa": ["büyükorhan", "harmancık", "keles"],
    "tekirdağ": ["saray", "kapaklı"],
    "balıkesir": ["kepsut", "dursunbey"],
    "çanakkale": ["bozcaada"],
    "edirne": ["enez", "ipsala"],
    "izmir": ["kiraz", "beydağ", "karaburun"],
    "manisa": ["demirci", "gördes"],
    "muğla": ["datça"],
    "afyonkarahisar": ["dinar", "çay", "sultandağı"],
    "ankara": ["elmadağ", "nallıhan", "beypazarı", "kazan", "kızılcahamam", "çamlıdere", "evren", "bala"],
    "eskişehir": ["inönü"],
    "konya": ["akşehir", "tuzlukçu", "ılgın", "doğanhisar"],
    "kayseri": ["yahyalı", "sarız"],
    "sivas": ["suşehri", "koyulhisar", "gölova", "akıncılar", "divriği"],
    "niğde": ["bor"],
    "çankırı": ["yapraklı"],
    "yozgat": ["akdağmadeni"],
    "antalya": ["demre", "finike", "kumluca", "kemer", "kaş", "alanya", "gündoğmuş", "gazipaşa"],
    "adana": ["tufanbeyli", "saimbeyli", "pozantı"],
    "mersin": ["tarsus", "çamlıyayla"],
    "kahramanmaraş": ["ekinözü"],
    "bolu": ["seben", "kıbrıscık"],
    "zonguldak": ["devrek", "gökçebey"],
    "bartın": ["ulus"],
    "kastamonu": ["tosya", "cide"],
    "sinop": ["boyabat", "durağan"],
    "samsun": ["ladik", "havza", "vezirköprü", "kavak", "bafra", "alaçam"],
    "ordu": ["mesudiye", "akkuş"],
    "giresun": ["alucra", "şebinkarahisar", "çamoluk"],
    "trabzon": ["of", "hayrat"],
    "artvin": ["yusufeli"],
    "tokat": ["artova"],
    "çorum": ["osmancık", "kargı", "mecitözü"],
    "tunceli": ["pülümür"],
    "diyarbakır": ["lice", "hani", "çermik"],
    "şanlıurfa": ["bozova"],
    "batman": ["sason"],
    "ağrı": ["patnos", "doğubayazıt"],
    "erzurum": ["ispir"],
    "kars": ["kağızman"],
    "ardahan": ["göle"]
}
KULLANIM_TURLERI = ["konut", "ofis", "okul", "hastane", "sanayi"]
KISA_KOLON_SECENEK = ["yok", "var", "emin_degil"]
AGIR_CIKMA_SECENEK = ["yok", "hafif", "buyuk"]
PLAN_TIPLERI = ["dikdortgen", "L", "T", "U", "kompleks"]
BITISIK_HIZA_SECENEK = ["yok", "uyumlu", "farkli"]
ZEMIN_SINIFLARI = ["Z1", "Z2", "Z3", "Z4"]


def rastgele_bina() -> RiskRequest:
    """Tek bir binanın form verisini rastgele üretir."""
    il = random.choice(list(ILLER_VE_ILCELER.keys()))
    ilce = random.choice(ILLER_VE_ILCELER[il])

    # Yapım yılı (1965–2024 arası)
    yapim_yili = random.randint(1965, 2024)

    # Kat sayısı (1–12 arası)
    kat_sayisi = random.randint(1, 12)

    # Zemin katta dükkân
    zemin_dukkan = random.choice(["evet", "hayir"])

    # Bitişik nizam
    bitisik = random.choice(["evet", "hayir"])

    # Gözle hasar
    hasar = random.choice(["yok", "hafif", "kolon"])

    kullanim_amaci = random.choice(KULLANIM_TURLERI)
    kisa_kolon = random.choice(KISA_KOLON_SECENEK)
    agir_cikma = random.choice(AGIR_CIKMA_SECENEK)
    plan_tipi = random.choice(PLAN_TIPLERI)
    bitisik_hiza = random.choice(BITISIK_HIZA_SECENEK)
    zemin_sinifi = random.choice(ZEMIN_SINIFLARI)

    # Backend'deki RiskRequest ile aynı alan isimlerini kullanıyoruz
    req = RiskRequest(
        il=il,
        ilce=ilce,
        yapimYili=yapim_yili,
        katSayisi=kat_sayisi,
        zeminDukkan=zemin_dukkan,
        bitisik=bitisik,
        hasar=hasar,
        kullanimAmaci=kullanim_amaci,
        kisaKolon=kisa_kolon,
        agirCikma=agir_cikma,
        planTipi=plan_tipi,
        bitisikHiza=bitisik_hiza,
        zeminSinifi=zemin_sinifi,
    )

    return req


def main():
    # CSV başlığı – train_model.py'dekiyle uyumlu
    header = [
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
        "gercekEtiket",  # Düşük / Orta / Yüksek
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for _ in range(KAYIT_SAYISI):
            req = rastgele_bina()

            # Deprem tehlikesi puanı
            deprem_seviye = deprem_seviyesi_bul(req.il, req.ilce)
            deprem_puan = deprem_seviyesi_puan(deprem_seviye)

            # Yapısal skor (backend'deki aynı fonksiyon)
            # main.py'deki fonksiyon imzası: yapisal_skor_hesapla(req, zemin_sinifi)
            yapisal_puan, _ = yapisal_skor_hesapla(req, req.zeminSinifi)


            # Toplam ve etiket
            toplam = deprem_puan + yapisal_puan
            etiket = genel_risk_seviyesi(toplam)  # "Düşük / Orta / Yüksek"

            writer.writerow(
                [
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
                    req.zeminSinifi,
                    etiket,
                ]
            )

    print(f"{KAYIT_SAYISI} satırlık sentetik_bina_verisi.csv oluşturuldu.")


if __name__ == "__main__":
    main()

