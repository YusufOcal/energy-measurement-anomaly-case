"""
ADIM 10: Analiz Sonuclarini Is Diline Cevir
Teknik degil, operasyonel ve yonetimsel perspektif.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("dataset_clean.csv", index_col=0)
df['load_profile_date'] = pd.to_datetime(df['load_profile_date'])

anomali_kolonlari = [
    'A1_Akim_Var_Tuketim_Yok', 'A2_Tuketim_Var_Akim_Dusuk',
    'A3_Faz_Dengesizligi_Yuksek', 'A4_Gerilim_Eksik_Tuketim_Var',
    'A5_Sifir_Negatif_Tuketim', 'A6_Sabit_Tuketim',
    'A7_Gece_Olagandisi', 'A8_Yuksek_Reaktif'
]

# Temel metrikler
toplam_kayit = len(df)
toplam_abone = df['tesisat_no_id'].nunique()
anomalili_kayit = (df['Toplam_Anomali_Sayisi'] > 0).sum()
anomalili_abone = df[df['Toplam_Anomali_Sayisi'] > 0]['tesisat_no_id'].nunique()

print("""
================================================================
    YEDAS ELEKTRIK OLCUM VERILERI
    IS ODAKLI ANALIZ RAPORU
================================================================

Analiz Kapsami:
  Donem       : Agustos - Eylul 2025 (61 gun)
  Veri        : 353,949 olcum kaydi, 15 dakikalik periyotlar
  Abone       : 74 OG/AG tesisat (5 il, 24 ilce)
  Anomali     : 8 farkli kural tabanli senaryo tanimlanmistir

================================================================
""")


# ============================================================
# 1. EN KRITIK 3 ANOMALI TURU
# ============================================================
print("=" * 64)
print("  1. EN KRITIK 3 ANOMALI TURU")
print("=" * 64)

a4_count = df['A4_Gerilim_Eksik_Tuketim_Var'].sum()
a4_abone = df[df['A4_Gerilim_Eksik_Tuketim_Var']]['tesisat_no_id'].nunique()

a8_count = df['A8_Yuksek_Reaktif'].sum()
a8_abone = df[df['A8_Yuksek_Reaktif']]['tesisat_no_id'].nunique()

a5_count = df['A5_Sifir_Negatif_Tuketim'].sum()
a5_abone = df[df['A5_Sifir_Negatif_Tuketim']]['tesisat_no_id'].nunique()
# Tamamen sifir olan abone
tamamen_sifir = df.groupby('tesisat_no_id')['A5_Sifir_Negatif_Tuketim'].mean()
tam_sifir_abone = (tamamen_sifir > 0.95).sum()

print(f"""
  KRITIK 1: SAYAC GERILIM OLCUM EKSIKLIGI
  ----------------------------------------
  Etki     : {a4_count:,} kayit ({a4_count/toplam_kayit*100:.1f}%) | {a4_abone} tesisat
  Ne oluyor: {a4_abone} tesisatta sayac enerji tuketimini kaydediyor
             ancak gerilim (voltaj) degeri raporlamiyor. Sayac
             calisiyor gibi gorunuyor ama olcum verisi eksik.
  Neden    : Tamami LUNA marka sayaclardan kaynaklidir.
             LUN23.5010 modeli gerilim sensoru icermiyor veya
             firmware seviyesinde bu veriyi iletmiyor.
  Neden    : Bu tesisatlarda gerilim tabanli herhangi bir
  Kritik     dogrulama veya kalite kontrolu yapilamiyor.
             Voltaj dusuklugunun tespit edilememesi, kayip-kacak
             incelemelerinde kor nokta olusturur.

  KRITIK 2: YUKSEK REAKTIF ENERJI TUKETIMI
  ----------------------------------------
  Etki     : {a8_count:,} kayit ({a8_count/toplam_kayit*100:.1f}%) | {a8_abone} tesisat
  Ne oluyor: {a8_abone} tesisatta reaktif enerji tuketimi, aktif
             tuketimin %62'sini asiyor (cos_phi < 0.85 esigi).
             Bu aboneler sebekeden gereksiz yere fazla guc cekiyor.
  Neden    : Kompanzasyon panolari ya kurulu degil, ya arizali,
             ya da yetersiz kapasitede calisiyor.
  Neden    : Reaktif enerji sebeke kayiplarini arttirir, trafo
  Kritik     ve kablolarda asiri isinmaya yol acar. Ayrica bu
             abonelere mevzuat geregi reaktif ceza uygulanmasi
             gerekmektedir. Tahakkuk eksikligi gelir kaybi
             anlamina gelir.

  KRITIK 3: SAYAC ILERLEMIYOR (SIFIR TUKETIM)
  ----------------------------------------
  Etki     : {a5_count:,} kayit ({a5_count/toplam_kayit*100:.1f}%) | {a5_abone} tesisat
  Ne oluyor: {a5_abone} tesisatta belirli periyotlarda sayac
             hicbir tuketim kaydetmiyor. Bunlardan {tam_sifir_abone} tanesi
             2 aylik donemde neredeyse hic ilerlememis.
  Neden    : Sayac donanimsal ariza, iletisim kopuklugu veya
             sayac bypass edilmis olabilir.
  Neden    : Sayac dururken tesisat enerji kullanmaya devam
  Kritik     ediyorsa, tuketim faturalandirilmiyor demektir.
             Bu dogrudan gelir kaybidir.
""")

# ============================================================
# 2. OLASI OPERASYONEL RISKLER
# ============================================================
print("=" * 64)
print("  2. OLASI OPERASYONEL RISKLER")
print("=" * 64)

# Hesaplamalar
a3_abone = df[df['A3_Faz_Dengesizligi_Yuksek']]['tesisat_no_id'].nunique()
a7_count = df['A7_Gece_Olagandisi'].sum()
a7_abone = df[df['A7_Gece_Olagandisi']]['tesisat_no_id'].nunique()

luna_abone = df[df['marka'] == 'LUNA']['tesisat_no_id'].nunique()
luna_toplam = df[df['marka'] == 'LUNA'].shape[0]

# Tahmini gelir kaybi (sifir tuketimli abonelerin ortalama tuketimine gore)
# Benzer profildeki calisan abonelerin tuketimi uzerinden tahmin
sifir_tesisatlar = tamamen_sifir[tamamen_sifir > 0.95].index
normal_tuketim_15dk = df[
    (df['Aktif_Tuketim_Farki'] > 0) & 
    (~df['tesisat_no_id'].isin(sifir_tesisatlar))
]['Aktif_Tuketim_Farki'].median()
# 2 ay, 96 periyot/gun * 61 gun
tahmini_kayip_kwh = normal_tuketim_15dk * 96 * 61 * tam_sifir_abone

print(f"""
  RISK 1: GELIR KAYBI
  -------------------
  - {tam_sifir_abone} tesisat 2 aydir tuketim kaydetmiyor.
  - Normal bir abonenin 15 dakikalik medyan tuketimi: {normal_tuketim_15dk:.3f} kWh
  - Tahmini 2 aylik faturalandirilmamis tuketim:
    yaklasik {tahmini_kayip_kwh:,.0f} kWh ({tam_sifir_abone} tesisat icin)
  - Reaktif ceza uygulanmayanlar da ayri bir gelir kaybi kalemi.

  RISK 2: EKIPMAN HASARI
  ----------------------
  - {a3_abone} tesisatta surekli faz dengesizligi tespit edildi.
  - 30 amper ustunde dengesizlik, trafo ve kablolarda asiri
    isinmaya neden olur. Notr kablosu asiri yuk altindadir.
  - Uzun vadede ekipman arizasi ve plansiz kesinti riski.

  RISK 3: KAYIP-KACAK KOR NOKTASI
  --------------------------------
  - {luna_abone} LUNA sayac ({luna_toplam:,} kayit) gerilim verisi
    uretmiyor. Bu abonelerde:
    > Dusuk gerilim kaynakli kayip hesaplanamaz
    > Voltaj manipulasyonu tespit edilemez
    > Enerji denklemi dogrulanamaz

  RISK 4: GECE SAATI ANORMALLIKLERI
  ---------------------------------
  - {a7_abone} tesisatta {a7_count:,} kez gece saatlerinde
    (22:00-06:00) normelin 3 katini asan tuketim tespit edildi.
  - Is saatleri disinda beklenmeyen yuksek tuketim, izinsiz
    kullanim veya kacak baglanti isareti olabilir.
""")

# ============================================================
# 3. SAHA EKIPLERI ICIN ONERILEN AKSIYONLAR
# ============================================================
print("=" * 64)
print("  3. SAHA EKIPLERI ICIN ONERILEN AKSIYONLAR")
print("=" * 64)

# En oncelikli tesisatlar
tesisat_risk = df.groupby('tesisat_no_id').agg(
    puan=('Toplam_Anomali_Sayisi', 'sum'),
    il=('il', 'first'),
    ilce=('ilce', 'first'),
    model=('model', 'first'),
    gerilim=('gerilim_seviyesi', 'first'),
).sort_values('puan', ascending=False)

top5 = tesisat_risk.head(5)

print(f"""
  AKSIYON 1: ACIL SAYAC KONTROLU (ONCELIK: YUKSEK)
  -------------------------------------------------
  Hedef  : Sifir tuketim gosteren tesisatlar
  Islem  : Saha ziyareti ile sayacin fiziksel durumu kontrol
           edilmeli. Sayac bypass, manyetik mudahale veya
           donanim arizasi arastirilmali.
  Abone  : {tam_sifir_abone} tesisat (A5 anomalisi %95+ olan)

  AKSIYON 2: LUNA SAYAC DEGISIM PROGRAMI (ONCELIK: YUKSEK)
  ---------------------------------------------------------
  Hedef  : LUNA LUN23.5010 ve LUN23-TF modelleri
  Islem  : Bu sayaclar gerilim ve/veya reaktif olcum yapmiyor.
           Veri butunlugu icin sayac degisimi veya en azindan
           firmware guncellemesi planlanmali.
  Abone  : {luna_abone} tesisat (tamami Samsun, Sinop, Amasya,
           Ordu ve Corum illerinde)

  AKSIYON 3: KOMPANZASYON PANOSU KONTROLU (ONCELIK: ORTA)
  --------------------------------------------------------
  Hedef  : Yuksek reaktif tuketimli tesisatlar (A8)
  Islem  : Mevcut kompanzasyon sisteminin kapasitesi, bakim
           durumu ve dogru calisip calismadigi kontrol edilmeli.
           Eksik ise kurulum onerilmeli.
  Abone  : {a8_abone} tesisat

  AKSIYON 4: FAZ DENGELEME (ONCELIK: ORTA)
  -----------------------------------------
  Hedef  : Surekli tek fazdan yuk ceken tesisatlar (A3)
  Islem  : Tesisat ic tesisatinda yuk dagilimi incelenmeli.
           Gerekirse uc faza dengeli dagitim yapilmali.
  Abone  : {a3_abone} tesisat

  AKSIYON 5: GECE DENETIMI (ONCELIK: DUSUK)
  ------------------------------------------
  Hedef  : Gece olagandisi tuketim gosteren tesisatlar (A7)
  Islem  : Is disinda yuksek tuketim gosterenlerin faaliyet
           durumu dogrulanmali. Kacak baglanti olasiligi
           degerlendirilmeli.
  Abone  : {a7_abone} tesisat

  ONCELIKLI ZIYARET LISTESI (ILK 5 TESISAT):
  -------------------------------------------""")

for i, (tesisat, row) in enumerate(top5.iterrows(), 1):
    print(f"    {i}. {tesisat[:20]}...")
    print(f"       {row['il']}/{row['ilce']} | {row['model']} | {row['gerilim']} | Risk puani: {int(row['puan']):,}")

# ============================================================
# 4. KAYIP-KACAK BIRIMI ICIN ONCELIKLENDIRME
# ============================================================
print(f"""

{'='*64}
  4. KAYIP-KACAK BIRIMI ICIN ONCELIKLENDIRME
{'='*64}

  ONCELIK MATRISI:

  ACIL (Bu hafta)
  ---------------
  > {tam_sifir_abone} tesisat sayaci 2 aydir ilerlemiyor. Faturalandirilmamis
    tuketim riski var. Saha kontrolu ile fiziksel dogrulama yapilmali.
  > Bu tesisatlarin elektrik tuketip tuketmedigini anlamak icin
    hat uzerinden akim olcumu alinmali.

  KISA VADE (Bu ay)
  -----------------
  > LUNA sayacli {luna_abone} tesisat icin gerilim verisi alinamiyor.
    Bu abonelerde voltaj tabanli kacak tespiti mumkun degil.
    Sayac degisim plani hazirlanmali.
  > {a8_abone} tesisatta reaktif ceza hesaplanmali ve gerekiyorsa
    tahakkuk surecine dahil edilmeli.

  ORTA VADE (Bu ceyrek)
  --------------------
  > Gece saatlerinde anormal tuketim gosteren {a7_abone} tesisat
    izlenmeye alinmali. Tekrarlayan paternler raporlanmali.
  > Faz dengesizligi yuksek {a3_abone} tesisat icin teknik
    duzeltme talebi olusturulmali.

  UZUN VADE (Stratejik)
  --------------------
  > LUNA marka sayac tedarik politikasi gozden gecirilmeli.
    Yeni alimlarda gerilim ve reaktif olcum zorunlu tutulmali.
  > 1 saatlik periyotla kayit tutan 13 abonenin sayaclari
    15 dakikalik periyoda donusturulmeli.
""")

# ============================================================
# 5. YONETIM ICIN 5 MADDELIK OZET KARAR CIKTISI
# ============================================================

# Ek metrikler
samsun_anomali = df[df['il'] == 'SAMSUN']['Toplam_Anomali_Sayisi'].gt(0).mean() * 100

print(f"""
{'='*64}
  5. YONETIM ICIN KARAR CIKTISI
{'='*64}

  +---------------------------------------------------------+
  |  YEDAS ELEKTRIK - OLCUM ANOMALI ANALIZI                 |
  |  YONETICI OZETI                                         |
  |  Donem: Agustos - Eylul 2025                            |
  +---------------------------------------------------------+

  1. 74 tesisatin %{anomalili_abone/toplam_abone*100:.0f}'inde en az bir anomali tespit
     edilmistir. Toplam kayitlarin %{anomalili_kayit/toplam_kayit*100:.0f}'si anomali icermektedir.
     Bunlarin buyuk cogunlugu sayac donanim eksikliginden
     kaynaklanmaktadir, sistematik bir sorundur.

  2. {luna_abone} LUNA marka sayac gerilim verisi uretmiyor.
     Bu tesisatlarda enerji denklemi dogrulanamaz, kayip-kacak
     tespiti yapilamaz. SAYAC DEGISIM PROGRAMI baslatilmasi
     oneriliyor.

  3. {tam_sifir_abone} tesisat 2 aydir hicbir tuketim kaydetmiyor.
     Tahmini faturalandirilmamis enerji: ~{tahmini_kayip_kwh:,.0f} kWh.
     ACIL SAHA KONTROLU gereklidir.

  4. {a8_abone} tesisatta reaktif enerji limitleri asilmistir
     (cos_phi < 0.85). Bu abonelere REAKTIF CEZA TAHAKKUKU
     baslatilmali ve kompanzasyon kurulumu saglanmalidir.

  5. En yuksek risk bolgesi SAMSUN'dur (%{samsun_anomali:.0f} anomali orani).
     Samsun bolge mudurlugu icin ozel bir SAYAC YENILEME ve
     SAHA DENETIM PLANI olusturulmasi oneriliyor.

  +---------------------------------------------------------+
""")
