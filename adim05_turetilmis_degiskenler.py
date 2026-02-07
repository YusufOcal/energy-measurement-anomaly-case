"""
ADIM 5: Turetilmis Degiskenler (Feature Engineering)
Temiz ve siralanmis veri uzerinden calisiyor (dataset_clean.csv).
"""

import pandas as pd
import numpy as np

df = pd.read_csv("dataset_clean.csv", index_col=0)
df['load_profile_date'] = pd.to_datetime(df['load_profile_date'])

print("Baslangic: {} satir, {} kolon".format(df.shape[0], df.shape[1]))

# ============================================================
# 1. Ortalama_Akim = (l1 + l2 + l3) / 3
# ============================================================
print("\n" + "=" * 60)
print("1. ORTALAMA_AKIM = (l1 + l2 + l3) / 3")
print("=" * 60)
print("Aciklama: Uc fazin akim degerlerinin aritmetik ortalamasidir.")
print("l1, l2, l3 kolonlarinda eksik deger yok, tum satirlar hesaplanir.")

df['Ortalama_Akim'] = (df['l1'] + df['l2'] + df['l3']) / 3

print(f"\n  Min    : {df['Ortalama_Akim'].min():.4f} A")
print(f"  Max    : {df['Ortalama_Akim'].max():.4f} A")
print(f"  Mean   : {df['Ortalama_Akim'].mean():.4f} A")
print(f"  Median : {df['Ortalama_Akim'].median():.4f} A")
print(f"  Null   : {df['Ortalama_Akim'].isnull().sum()}")

print(f"\n  Ornek satirlar:")
ornekler = df[['l1', 'l2', 'l3', 'Ortalama_Akim']].head(5)
print(f"  {'l1':>8} {'l2':>8} {'l3':>8} {'Ort_Akim':>10}")
print(f"  {'-'*38}")
for _, row in ornekler.iterrows():
    print(f"  {row['l1']:>8.2f} {row['l2']:>8.2f} {row['l3']:>8.2f} {row['Ortalama_Akim']:>10.4f}")

# ============================================================
# 2. Ortalama_Gerilim = (v1 + v2 + v3) / 3
# ============================================================
print("\n" + "=" * 60)
print("2. ORTALAMA_GERILIM = (v1 + v2 + v3) / 3")
print("=" * 60)
print("Aciklama: Uc fazin gerilim degerlerinin aritmetik ortalamasidir.")
print("v1/v2/v3 eksik olan satirlarda sonuc NaN olur (propagate).")

df['Ortalama_Gerilim'] = (df['v1'] + df['v2'] + df['v3']) / 3

dolu = df['Ortalama_Gerilim'].notna().sum()
eksik = df['Ortalama_Gerilim'].isnull().sum()
print(f"\n  Dolu   : {dolu:,}")
print(f"  Null   : {eksik:,} (v1/v2/v3 eksik olan satirlar)")
print(f"  Min    : {df['Ortalama_Gerilim'].min():.4f} V")
print(f"  Max    : {df['Ortalama_Gerilim'].max():.4f} V")
print(f"  Mean   : {df['Ortalama_Gerilim'].mean():.4f} V")
print(f"  Median : {df['Ortalama_Gerilim'].median():.4f} V")

print(f"\n  Ornek satirlar (dolu olanlar):")
orn_v = df[df['Ortalama_Gerilim'].notna()][['v1', 'v2', 'v3', 'Ortalama_Gerilim']].head(5)
print(f"  {'v1':>8} {'v2':>8} {'v3':>8} {'Ort_Ger':>10}")
print(f"  {'-'*38}")
for _, row in orn_v.iterrows():
    print(f"  {row['v1']:>8.2f} {row['v2']:>8.2f} {row['v3']:>8.2f} {row['Ortalama_Gerilim']:>10.4f}")

print(f"\n  Ornek satirlar (eksik olanlar):")
orn_vn = df[df['Ortalama_Gerilim'].isna()][['v1', 'v2', 'v3', 'Ortalama_Gerilim']].head(3)
print(f"  {'v1':>8} {'v2':>8} {'v3':>8} {'Ort_Ger':>10}")
print(f"  {'-'*38}")
for _, row in orn_vn.iterrows():
    v1 = f"{row['v1']:.2f}" if pd.notna(row['v1']) else "NaN"
    v2 = f"{row['v2']:.2f}" if pd.notna(row['v2']) else "NaN"
    v3 = f"{row['v3']:.2f}" if pd.notna(row['v3']) else "NaN"
    print(f"  {v1:>8} {v2:>8} {v3:>8} {'NaN':>10}")

# ============================================================
# 3. Faz_Dengesizligi = max(l1,l2,l3) - min(l1,l2,l3)
# ============================================================
print("\n" + "=" * 60)
print("3. FAZ_DENGESIZLIGI = max(l1,l2,l3) - min(l1,l2,l3)")
print("=" * 60)
print("Aciklama: Uc faz arasindaki en buyuk ve en kucuk akim farkidir.")
print("Deger buyudukce faz yuklemesi dengesizdir. Sifir = tam dengeli.")

df['Faz_Dengesizligi'] = df[['l1', 'l2', 'l3']].max(axis=1) - df[['l1', 'l2', 'l3']].min(axis=1)

print(f"\n  Min    : {df['Faz_Dengesizligi'].min():.4f} A")
print(f"  Max    : {df['Faz_Dengesizligi'].max():.4f} A")
print(f"  Mean   : {df['Faz_Dengesizligi'].mean():.4f} A")
print(f"  Median : {df['Faz_Dengesizligi'].median():.4f} A")
print(f"  Null   : {df['Faz_Dengesizligi'].isnull().sum()}")
print(f"  Sifir  : {(df['Faz_Dengesizligi'] == 0).sum():,} (tam dengeli anlar)")

print(f"\n  Ornek satirlar:")
orn_fd = df[['l1', 'l2', 'l3', 'Faz_Dengesizligi']].head(5)
print(f"  {'l1':>8} {'l2':>8} {'l3':>8} {'Faz_Deng':>10}")
print(f"  {'-'*38}")
for _, row in orn_fd.iterrows():
    print(f"  {row['l1']:>8.2f} {row['l2']:>8.2f} {row['l3']:>8.2f} {row['Faz_Dengesizligi']:>10.4f}")

# ============================================================
# 4. Aktif_Tuketim_Farki = t0 - onceki t0 (tesisat bazinda)
# ============================================================
print("\n" + "=" * 60)
print("4. AKTIF_TUKETIM_FARKI = t0(n) - t0(n-1)  [tesisat bazinda]")
print("=" * 60)
print("Aciklama: t0 kumulatif sayac degeridir. Her abonenin ardisik iki")
print("olcumu arasindaki fark o periyottaki enerji tuketimini verir.")
print("Veri onceden tesisat+zaman bazinda siralanmistir.")
print("Her abonenin ilk kaydi icin NaN olusur.")

df['Aktif_Tuketim_Farki'] = df.groupby('tesisat_no_id')['t0'].diff()

dolu_atf = df['Aktif_Tuketim_Farki'].notna().sum()
eksik_atf = df['Aktif_Tuketim_Farki'].isnull().sum()
negatif_atf = (df['Aktif_Tuketim_Farki'] < 0).sum()
sifir_atf = (df['Aktif_Tuketim_Farki'] == 0).sum()

print(f"\n  Dolu     : {dolu_atf:,}")
print(f"  Null     : {eksik_atf:,} (her abonenin ilk kaydi = 74 abone)")
print(f"  Negatif  : {negatif_atf:,}")
print(f"  Sifir    : {sifir_atf:,}")

atf = df['Aktif_Tuketim_Farki'].dropna()
print(f"  Min      : {atf.min():.4f} kWh")
print(f"  Max      : {atf.max():.4f} kWh")
print(f"  Mean     : {atf.mean():.4f} kWh")
print(f"  Median   : {atf.median():.4f} kWh")

print(f"\n  Ornek satirlar (ayni abonenin ardisik kayitlari):")
ilk_tesisat = df['tesisat_no_id'].iloc[0]
orn_atf = df[df['tesisat_no_id'] == ilk_tesisat][['load_profile_date', 't0', 'Aktif_Tuketim_Farki']].head(6)
print(f"  Abone: {ilk_tesisat[:12]}...")
print(f"  {'Tarih':>23} {'t0':>12} {'Fark':>10}")
print(f"  {'-'*47}")
for _, row in orn_atf.iterrows():
    fark = f"{row['Aktif_Tuketim_Farki']:.4f}" if pd.notna(row['Aktif_Tuketim_Farki']) else "NaN"
    print(f"  {str(row['load_profile_date']):>23} {row['t0']:>12.4f} {fark:>10}")

# ============================================================
# 5. Gerilim_Sapma_Orani
# ============================================================
print("\n" + "=" * 60)
print("5. GERILIM_SAPMA_ORANI")
print("=" * 60)
print("Aciklama: Ortalama gerilimin nominal 230V'tan yuzde sapmasidir.")
print("Formul: |Ortalama_Gerilim - 230| / 230 * 100")
print("Turkiye standardi 230V +/-%10 -> esik %10.")
print("v1/v2/v3 eksik olan satirlarda NaN olur.")

nominal = 230
df['Gerilim_Sapma_Orani'] = (abs(df['Ortalama_Gerilim'] - nominal) / nominal * 100)

dolu_gs = df['Gerilim_Sapma_Orani'].notna().sum()
print(f"\n  Dolu   : {dolu_gs:,}")
print(f"  Null   : {df['Gerilim_Sapma_Orani'].isnull().sum():,}")

gs = df['Gerilim_Sapma_Orani'].dropna()
print(f"  Min    : {gs.min():.4f}%")
print(f"  Max    : {gs.max():.4f}%")
print(f"  Mean   : {gs.mean():.4f}%")
print(f"  Median : {gs.median():.4f}%")
print(f"  >%10   : {(gs > 10).sum():,} ({(gs > 10).mean()*100:.2f}%)")
print(f"  >%20   : {(gs > 20).sum():,} ({(gs > 20).mean()*100:.2f}%)")

print(f"\n  Ornek satirlar:")
orn_gs = df[df['Gerilim_Sapma_Orani'].notna()][['Ortalama_Gerilim', 'Gerilim_Sapma_Orani']].head(5)
print(f"  {'Ort_Gerilim':>12} {'Sapma%':>10}")
print(f"  {'-'*24}")
for _, row in orn_gs.iterrows():
    print(f"  {row['Ortalama_Gerilim']:>12.4f} {row['Gerilim_Sapma_Orani']:>10.4f}")

# ============================================================
# 6. Saat_Dilimi: Gece / Gunduz / Mesai
# ============================================================
print("\n" + "=" * 60)
print("6. SAAT_DILIMI: Gece / Gunduz / Mesai")
print("=" * 60)
print("Aciklama: Saat bilgisine gore 3 dilim tanimlanir.")
print("  Gece  : 22:00 - 05:59 (dusuk tuketim, gece tarife)")
print("  Mesai : 06:00 - 17:59 (is saatleri, yuksek tuketim)")
print("  Gunduz: 18:00 - 21:59 (aksam puant, orta tuketim)")

saat = df['load_profile_date'].dt.hour

df['Saat_Dilimi'] = np.where(
    (saat >= 22) | (saat <= 5), 'Gece',
    np.where(
        (saat >= 6) & (saat <= 17), 'Mesai',
        'Gunduz'  # 18-21
    )
)

dilim_dist = df['Saat_Dilimi'].value_counts()
print(f"\n  Dagilim:")
for dilim, sayi in dilim_dist.items():
    print(f"    {dilim:<8}: {sayi:,} kayit ({sayi/len(df)*100:.1f}%)")
print(f"  Null: {df['Saat_Dilimi'].isnull().sum()}")

print(f"\n  Ornek satirlar:")
orn_sd = df[['load_profile_date', 'Saat_Dilimi']].iloc[::40].head(8)
print(f"  {'Tarih-Saat':>23} {'Saat':>5} {'Dilim':>8}")
print(f"  {'-'*38}")
for _, row in orn_sd.iterrows():
    s = row['load_profile_date'].hour
    print(f"  {str(row['load_profile_date']):>23} {s:>5} {row['Saat_Dilimi']:>8}")

# ============================================================
# OZET
# ============================================================
print("\n" + "=" * 60)
print("OZET: TURETILMIS DEGISKENLER")
print("=" * 60)

yeni_kolonlar = ['Ortalama_Akim', 'Ortalama_Gerilim', 'Faz_Dengesizligi', 
                 'Aktif_Tuketim_Farki', 'Gerilim_Sapma_Orani', 'Saat_Dilimi']

print(f"\n  {'Kolon':<25} {'Tip':<12} {'Dolu':>10} {'Null':>10}")
print(f"  {'-'*57}")
for col in yeni_kolonlar:
    dolu = df[col].notna().sum()
    eksik = df[col].isnull().sum()
    tip = str(df[col].dtype)
    print(f"  {col:<25} {tip:<12} {dolu:>10,} {eksik:>10,}")

print(f"\n  Toplam kolon: 18 (orijinal) + 6 (yeni) = {df.shape[1]}")
print(f"  Satir sayisi: {df.shape[0]:,} (degismedi)")

# Kaydet
df.to_csv("dataset_clean.csv")
print(f"\n  Kaydedildi: dataset_clean.csv")
