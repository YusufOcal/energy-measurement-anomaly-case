"""
ADIM 7: Anomali Kurallarini Koda Dok
Her anomali icin boolean kolon + Toplam_Anomali_Sayisi ozet kolonu.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("dataset_clean.csv", index_col=0)
df['load_profile_date'] = pd.to_datetime(df['load_profile_date'])

print(f"Baslangic: {df.shape[0]:,} satir, {df.shape[1]} kolon")


# ============================================================
# ANOMALI FONKSIYONLARI (moduler yapi)
# ============================================================

def anomali_1_akim_var_tuketim_yok(df):
    """Ortalama_Akim > 1A VE Aktif_Tuketim_Farki == 0"""
    return (df['Ortalama_Akim'] > 1) & (df['Aktif_Tuketim_Farki'] == 0)


def anomali_2_tuketim_var_akim_dusuk(df):
    """Aktif_Tuketim_Farki > 1 kWh VE Ortalama_Akim < 0.5A"""
    return (df['Aktif_Tuketim_Farki'] > 1) & (df['Ortalama_Akim'] < 0.5)


def anomali_3_faz_dengesizligi_yuksek_akim(df):
    """Faz_Dengesizligi > 30A VE Ortalama_Akim > 10A"""
    return (df['Faz_Dengesizligi'] > 30) & (df['Ortalama_Akim'] > 10)


def anomali_4_gerilim_eksik_tuketim_var(df):
    """Ortalama_Gerilim NaN VE Aktif_Tuketim_Farki > 0"""
    return df['Ortalama_Gerilim'].isna() & (df['Aktif_Tuketim_Farki'] > 0)


def anomali_5_sifir_negatif_tuketim(df):
    """Aktif_Tuketim_Farki <= 0 (NaN haric)"""
    return df['Aktif_Tuketim_Farki'].notna() & (df['Aktif_Tuketim_Farki'] <= 0)


def anomali_6_sabit_tuketim(df, esik=8):
    """Ayni tesisat icinde ardisik >= esik kayit ayni delta_t0 degerine sahip (>0)"""
    sonuc = pd.Series(False, index=df.index)
    
    for tesisat in df['tesisat_no_id'].unique():
        mask_t = df['tesisat_no_id'] == tesisat
        sub = df.loc[mask_t, 'Aktif_Tuketim_Farki']
        
        # Sadece pozitif degerler uzerinde
        pozitif = sub > 0
        # Deger degisim noktalari
        degisim = sub.ne(sub.shift()) | (~pozitif)
        grup = degisim.cumsum()
        
        # Grup uzunluklari
        for g, grp in sub[pozitif].groupby(grup[pozitif]):
            if len(grp) >= esik:
                sonuc.loc[grp.index] = True
    
    return sonuc


def anomali_7_gece_olagandisi(df):
    """Gece diliminde, abonenin kendi gece ortalamasinin 3 katindan fazla tuketim"""
    sonuc = pd.Series(False, index=df.index)
    
    gece_mask = df['Saat_Dilimi'] == 'Gece'
    gece_pozitif = gece_mask & (df['Aktif_Tuketim_Farki'] > 0)
    
    # Her abonenin gece ortalamasini hesapla
    abone_gece_ort = df[gece_pozitif].groupby('tesisat_no_id')['Aktif_Tuketim_Farki'].mean()
    
    for tesisat, gece_ort in abone_gece_ort.items():
        if gece_ort <= 0:
            continue
        esik = gece_ort * 3
        mask = (
            (df['tesisat_no_id'] == tesisat) & 
            gece_mask & 
            (df['Aktif_Tuketim_Farki'] > esik)
        )
        sonuc.loc[mask] = True
    
    return sonuc


def anomali_8_yuksek_reaktif(df):
    """delta_ri / delta_t0 > 0.62 (cos_phi < 0.85)"""
    sonuc = pd.Series(False, index=df.index)
    
    # delta_ri hesapla
    delta_ri = df.groupby('tesisat_no_id')['ri'].diff()
    
    valid = delta_ri.notna() & (delta_ri > 0) & (df['Aktif_Tuketim_Farki'] > 0)
    oran = delta_ri / df['Aktif_Tuketim_Farki']
    
    sonuc.loc[valid & (oran > 0.62)] = True
    
    return sonuc


# ============================================================
# 1. HER ANOMALI ICIN BOOLEAN KOLON OLUSTUR
# ============================================================
print("\n" + "=" * 60)
print("1. ANOMALI KOLONLARI OLUSTURULUYOR")
print("=" * 60)

anomali_tanimlari = [
    ("A1_Akim_Var_Tuketim_Yok",       anomali_1_akim_var_tuketim_yok),
    ("A2_Tuketim_Var_Akim_Dusuk",      anomali_2_tuketim_var_akim_dusuk),
    ("A3_Faz_Dengesizligi_Yuksek",     anomali_3_faz_dengesizligi_yuksek_akim),
    ("A4_Gerilim_Eksik_Tuketim_Var",   anomali_4_gerilim_eksik_tuketim_var),
    ("A5_Sifir_Negatif_Tuketim",       anomali_5_sifir_negatif_tuketim),
    ("A6_Sabit_Tuketim",               anomali_6_sabit_tuketim),
    ("A7_Gece_Olagandisi",             anomali_7_gece_olagandisi),
    ("A8_Yuksek_Reaktif",              anomali_8_yuksek_reaktif),
]

for kolon_adi, fonksiyon in anomali_tanimlari:
    print(f"  Hesaplaniyor: {kolon_adi}...", end=" ")
    df[kolon_adi] = fonksiyon(df)
    true_count = df[kolon_adi].sum()
    print(f"-> {true_count:,} True")


# ============================================================
# 2. TOPLAM_ANOMALI_SAYISI OZET KOLON
# ============================================================
print("\n" + "=" * 60)
print("2. TOPLAM_ANOMALI_SAYISI KOLONU")
print("=" * 60)

anomali_kolonlari = [ad for ad, _ in anomali_tanimlari]
df['Toplam_Anomali_Sayisi'] = df[anomali_kolonlari].sum(axis=1)

print(f"  Min: {df['Toplam_Anomali_Sayisi'].min()}")
print(f"  Max: {df['Toplam_Anomali_Sayisi'].max()}")
print(f"  Mean: {df['Toplam_Anomali_Sayisi'].mean():.4f}")

# Dagilim
print(f"\n  Anomali sayisi dagilimi:")
dist = df['Toplam_Anomali_Sayisi'].value_counts().sort_index()
for sayi, kayit in dist.items():
    pct = kayit / len(df) * 100
    print(f"    {sayi} anomali : {kayit:,} satir ({pct:.2f}%)")


# ============================================================
# 3. BIR SATIR BIRDEN FAZLA ANOMALI ICEREBILIR (dogrulama)
# ============================================================
print("\n" + "=" * 60)
print("3. COK ANOMALILI SATIRLAR")
print("=" * 60)

coklu = df[df['Toplam_Anomali_Sayisi'] >= 2]
print(f"  2+ anomali iceren satir: {len(coklu):,}")
print(f"  3+ anomali iceren satir: {(df['Toplam_Anomali_Sayisi'] >= 3).sum():,}")

if len(coklu) > 0:
    print(f"\n  En cok anomali iceren satirlardan ornekler:")
    en_cok = df.nlargest(5, 'Toplam_Anomali_Sayisi')
    for _, row in en_cok.iterrows():
        aktif_anomaliler = [ad for ad in anomali_kolonlari if row[ad]]
        print(f"    {row['tesisat_no_id'][:16]}... | {row['load_profile_date']} | "
              f"Anomali: {int(row['Toplam_Anomali_Sayisi'])} -> {', '.join(aktif_anomaliler)}")

# Anomali capraz tablosu (en sik birlikte gorulen cifler)
print(f"\n  Anomali es-gorunum matrisi (kayit sayisi):")
co_matrix = df[anomali_kolonlari].astype(int).T.dot(df[anomali_kolonlari].astype(int))
# Sadece ust ucgen (capraz haric)
print(f"  {'':>6}", end="")
for i, col in enumerate(anomali_kolonlari):
    print(f" {col.split('_')[0]:>5}", end="")
print()
for i, row_name in enumerate(anomali_kolonlari):
    print(f"  {row_name.split('_')[0]:>6}", end="")
    for j, col_name in enumerate(anomali_kolonlari):
        if j < i:
            print(f"     -", end="")
        else:
            print(f" {co_matrix.iloc[i, j]:>5}", end="")
    print()


# ============================================================
# 4. ANOMALI KOLONLARININ DOLULUK VE DAGILIMI
# ============================================================
print("\n" + "=" * 60)
print("4. ANOMALI KOLONLARI DOLULUK VE DAGILIM RAPORU")
print("=" * 60)

print(f"\n  {'Kolon':<35} {'True':>8} {'False':>8} {'True%':>8} {'Abone':>6}")
print(f"  {'-'*67}")

for kolon_adi in anomali_kolonlari:
    true_c = df[kolon_adi].sum()
    false_c = (~df[kolon_adi]).sum()
    pct = true_c / len(df) * 100
    abone = df[df[kolon_adi]]['tesisat_no_id'].nunique() if true_c > 0 else 0
    print(f"  {kolon_adi:<35} {true_c:>8,} {false_c:>8,} {pct:>7.2f}% {abone:>6}")

# Toplam anomali ozeti
herhangi_anomali = (df['Toplam_Anomali_Sayisi'] > 0).sum()
temiz_satir = (df['Toplam_Anomali_Sayisi'] == 0).sum()
print(f"\n  {'Herhangi anomali (>=1)':<35} {herhangi_anomali:>8,} {'':>8} {herhangi_anomali/len(df)*100:>7.2f}%")
print(f"  {'Temiz satir (0 anomali)':<35} {temiz_satir:>8,} {'':>8} {temiz_satir/len(df)*100:>7.2f}%")
print(f"  {'TOPLAM':<35} {len(df):>8,}")

# Abone bazinda anomali yogunlugu
print(f"\n  Abone bazinda anomali yogunlugu:")
abone_anomali = df.groupby('tesisat_no_id')['Toplam_Anomali_Sayisi'].agg(
    toplam_anomali='sum',
    kayit='size',
    anomalili_kayit=lambda x: (x > 0).sum()
)
abone_anomali['oran'] = (abone_anomali['anomalili_kayit'] / abone_anomali['kayit'] * 100).round(1)
abone_anomali = abone_anomali.sort_values('oran', ascending=False)

print(f"  {'Abone':<20} {'Anomalili':>10} {'Toplam':>8} {'Oran':>8}")
print(f"  {'-'*48}")
for idx, row in abone_anomali.head(15).iterrows():
    gs = df[df['tesisat_no_id'] == idx]['gerilim_seviyesi'].iloc[0]
    print(f"  {idx[:18]}.. {int(row['anomalili_kayit']):>10} {int(row['kayit']):>8} {row['oran']:>7.1f}% {gs}")


# ============================================================
# FINAL: Kolon listesi ve kaydet
# ============================================================
print("\n" + "=" * 60)
print("FINAL: KOLON LISTESI")
print("=" * 60)
print(f"\n  Toplam kolon: {df.shape[1]}")
print(f"  Yeni eklenen: {len(anomali_kolonlari) + 1} (8 anomali + 1 ozet)")
print(f"\n  Tum kolonlar:")
for i, col in enumerate(df.columns, 1):
    tip = str(df[col].dtype)
    yeni = " [YENI]" if col in anomali_kolonlari or col == 'Toplam_Anomali_Sayisi' else ""
    print(f"    {i:>2}. {col:<35} {tip:<12}{yeni}")

df.to_csv("dataset_clean.csv")
print(f"\nKaydedildi: dataset_clean.csv ({df.shape[0]:,} satir, {df.shape[1]} kolon)")
