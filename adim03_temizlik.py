"""
ADIM 3: Temizlik Stratejisini Uygula
Onceki adimda belirlenen kararlarin birebir uygulanmasi.
"""

import pandas as pd
import numpy as np

# ============================================================
# TEMIZLIK ONCESI DURUM
# ============================================================
df = pd.read_csv("dataset.csv", index_col=0)

print("=" * 60)
print("TEMIZLIK ONCESI DURUM")
print("=" * 60)
print(f"Satir: {df.shape[0]:,} | Kolon: {df.shape[1]}")
print(f"Eksik deger toplam: {df.isnull().sum().sum():,}")

# ============================================================
# 1. t0: Negatif veya sifir kayitlar
# ============================================================
print("\n" + "=" * 60)
print("1. t0 KONTROLU VE ISLEMI")
print("=" * 60)

negatif_t0 = (df['t0'] < 0).sum()
sifir_t0 = (df['t0'] == 0).sum()
print(f"Negatif t0: {negatif_t0}")
print(f"Sifir t0  : {sifir_t0}")
print(f"Islem     : Silinecek veya isaretlenecek kayit YOK.")
print(f"            Tum t0 degerleri pozitif (min={df['t0'].min():.4f}).")

# ============================================================
# 2. v1, v2, v3: Eksik deger isleme
# ============================================================
print("\n" + "=" * 60)
print("2. GERILIM (v1, v2, v3) EKSIK DEGER ISLEME")
print("=" * 60)

v_eksik_oncesi = df['v1'].isnull().sum()
print(f"Eksik satir (oncesi): {v_eksik_oncesi:,} ({v_eksik_oncesi/len(df)*100:.2f}%)")

# Strateji: Satirlari silmiyoruz, voltaj analizinde filtrelenecek.
# Ancak dolu satirlardaki sifir voltaj degerlerini NaN'a ceviriyoruz
# cunku 0V fiziksel olarak anlamsiz (olcum hatasi veya kesinti).
v_sifir_oncesi = {}
for col in ['v1', 'v2', 'v3']:
    sifir_sayisi = ((df[col] == 0) & df[col].notna()).sum()
    v_sifir_oncesi[col] = sifir_sayisi
    df.loc[(df[col] == 0) & df[col].notna(), col] = np.nan
    print(f"  {col}: {sifir_sayisi:,} sifir deger -> NaN'a cevirildi")

v_eksik_sonrasi = df['v1'].isnull().sum()
print(f"\nEksik satir (sonrasi): {v_eksik_sonrasi:,} ({v_eksik_sonrasi/len(df)*100:.2f}%)")
print(f"Eklenen NaN: {v_eksik_sonrasi - v_eksik_oncesi:,}")
print(f"Satir silme : YAPILMADI (diger analizler icin korunuyor)")

# ============================================================
# 3. ri, rc: Eksik deger isleme
# ============================================================
print("\n" + "=" * 60)
print("3. REAKTIF ENERJI (ri, rc) EKSIK DEGER ISLEME")
print("=" * 60)

ri_eksik_oncesi = df['ri'].isnull().sum()
print(f"Eksik satir (oncesi): {ri_eksik_oncesi:,} ({ri_eksik_oncesi/len(df)*100:.2f}%)")
print(f"Islem: Satirlar silmiyoruz, reaktif analizde filtrelenecek.")
print(f"       ri/rc eksik olan 7 abone (LUN23-TF) farkli analizlerde korunuyor.")
print(f"Satir silme : YAPILMADI")

# ============================================================
# 4. Veri tipi dogrulamalari ve donusumleri
# ============================================================
print("\n" + "=" * 60)
print("4. VERI TIPI DOGRULAMA VE DONUSUMLERI")
print("=" * 60)

# 4a: load_profile_date -> datetime
print("\n  [load_profile_date]")
print(f"    Oncesi : {df['load_profile_date'].dtype}")
df['load_profile_date'] = pd.to_datetime(df['load_profile_date'])
print(f"    Sonrasi: {df['load_profile_date'].dtype}")

# 4b: Sayisal kolonlari dogrula
print("\n  Sayisal kolon dogrulamasi:")
sayisal_kolonlar = {
    'son_carpan_degeri': 'int64',
    'l1': 'float64',
    'l2': 'float64',
    'l3': 'float64',
    'v1': 'float64',
    'v2': 'float64',
    'v3': 'float64',
    't0': 'float64',
    'ri': 'float64',
    'rc': 'float64'
}

for col, beklenen in sayisal_kolonlar.items():
    mevcut = str(df[col].dtype)
    uyum = "OK" if mevcut == beklenen else "UYUMSUZ"
    print(f"    {col:<20} beklenen: {beklenen:<10} mevcut: {mevcut:<10} -> {uyum}")

# 4c: Kategorik kolonlar
print("\n  Kategorik kolon tipleri:")
kategorik_kolonlar = ['tesisat_no_id', 'il', 'ilce', 'gerilim_seviyesi', 'marka', 'model', 'abone_grubu']
for col in kategorik_kolonlar:
    print(f"    {col:<20} {str(df[col].dtype):<10} tekil: {df[col].nunique()}")

# ============================================================
# 5. Temizlik sonrasi rapor
# ============================================================
print("\n" + "=" * 60)
print("5. TEMIZLIK SONRASI RAPOR")
print("=" * 60)

print(f"\n  Satir sayisi:")
print(f"    Oncesi  : 353,949")
print(f"    Sonrasi : {df.shape[0]:,}")
print(f"    Silinen : {353949 - df.shape[0]:,}")

print(f"\n  Kolon sayisi:")
print(f"    Oncesi  : 18")
print(f"    Sonrasi : {df.shape[1]}")

print(f"\n  Eksik deger durumu (kolon bazinda):")
print(f"    {'Kolon':<25} {'Oncesi':>10} {'Sonrasi':>10} {'Degisim':>10}")
print(f"    {'-'*55}")

oncesi_eksik = {
    'tesisat_no_id': 0, 'il': 0, 'ilce': 0, 'gerilim_seviyesi': 0,
    'marka': 0, 'model': 0, 'abone_grubu': 0, 'son_carpan_degeri': 0,
    'l1': 0, 'l2': 0, 'l3': 0,
    'v1': 75434, 'v2': 75434, 'v3': 75434,
    't0': 0, 'ri': 10240, 'rc': 10240,
    'load_profile_date': 0
}

for col in df.columns:
    oncesi = oncesi_eksik.get(col, 0)
    sonrasi = df[col].isnull().sum()
    degisim = sonrasi - oncesi
    degisim_str = f"+{degisim:,}" if degisim > 0 else f"{degisim:,}" if degisim < 0 else "0"
    if sonrasi > 0 or oncesi > 0:
        print(f"    {col:<25} {oncesi:>10,} {sonrasi:>10,} {degisim_str:>10}")

print(f"\n    {'TOPLAM':<25} {sum(oncesi_eksik.values()):>10,} {df.isnull().sum().sum():>10,}")

print(f"\n  Yapilan islemler ozeti:")
print(f"    [1] t0: Islem gerekmedi (negatif/sifir yok)")
print(f"    [2] v1/v2/v3: 0V degerler NaN'a cevirildi (v1:+4408, v2:+2945, v3:+58)")
print(f"    [3] ri/rc: Degisiklik yapilmadi (analizde filtrelenecek)")
print(f"    [4] load_profile_date: object -> datetime64[ns] donusumu yapildi")
print(f"    [5] Satir silme: YAPILMADI (tum satirlar korundu)")

# Temiz veriyi kaydet
df.to_csv("dataset_clean.csv")
print(f"\n  Temiz veri kaydedildi: dataset_clean.csv")
print(f"  Veri analiz icin hazir.")
