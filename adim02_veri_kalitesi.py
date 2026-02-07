"""
ADIM 2: Veri Kalitesi Degerlendirmesi ve Temizlik Plani
Sadece analiz - hicbir degisiklik yapilmiyor.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv", index_col=0)

# ============================================================
# 1. Negatif veya sifir enerji tuketimi (t0) var mi?
# ============================================================
print("=" * 60)
print("1. ENERJI TUKETIMI (t0) KONTROLU")
print("=" * 60)

print(f"t0 temel istatistikler:")
print(f"  Min    : {df['t0'].min():.4f}")
print(f"  Max    : {df['t0'].max():.4f}")
print(f"  Mean   : {df['t0'].mean():.4f}")
print(f"  Median : {df['t0'].median():.4f}")

negatif_t0 = (df['t0'] < 0).sum()
sifir_t0 = (df['t0'] == 0).sum()
pozitif_t0 = (df['t0'] > 0).sum()

print(f"\nDeger dagilimi:")
print(f"  Negatif (t0 < 0)  : {negatif_t0:,}")
print(f"  Sifir   (t0 == 0) : {sifir_t0:,}")
print(f"  Pozitif (t0 > 0)  : {pozitif_t0:,}")
print(f"  Null              : {df['t0'].isnull().sum():,}")

# t0 kumulatif bir sayac degeri - periyodik fark (delta) icin siralama gerekli
# Simdilik sadece ham t0'a bakiyoruz
# Dusuk t0 degerleri (sayac baslangici olabilir)
print(f"\nEn dusuk 10 t0 degeri:")
lowest = df.nsmallest(10, 't0')[['tesisat_no_id', 't0', 'load_profile_date']]
for _, row in lowest.iterrows():
    print(f"  {row['tesisat_no_id'][:12]}... | t0={row['t0']:.4f} | {row['load_profile_date']}")

# ============================================================
# 2. Akim degerleri (l1, l2, l3) - mantiksiz degerler
# ============================================================
print("\n" + "=" * 60)
print("2. AKIM DEGERLERI (l1, l2, l3) KONTROLU")
print("=" * 60)

for col in ['l1', 'l2', 'l3']:
    s = df[col]
    print(f"\n  [{col}]")
    print(f"    Min: {s.min():.4f} | Max: {s.max():.4f} | Mean: {s.mean():.4f} | Median: {s.median():.4f}")
    print(f"    Null       : {s.isnull().sum():,}")
    print(f"    Negatif    : {(s < 0).sum():,}")
    print(f"    Sifir      : {(s == 0).sum():,} ({(s == 0).mean()*100:.2f}%)")
    print(f"    > 100A     : {(s > 100).sum():,}")
    print(f"    > 150A     : {(s > 150).sum():,}")

# Uc deger detayi
print(f"\n  En yuksek akim degerleri (l1, l2, l3 ayri ayri):")
for col in ['l1', 'l2', 'l3']:
    top5 = df.nlargest(5, col)[[col, 'tesisat_no_id', 'gerilim_seviyesi', 'load_profile_date']]
    print(f"\n    {col} - En yuksek 5:")
    for _, row in top5.iterrows():
        print(f"      {row[col]:.2f}A | {row['tesisat_no_id'][:12]}... | {row['gerilim_seviyesi']} | {row['load_profile_date']}")

# Uc fazin hepsi sifir olan satirlar
hepsi_sifir = ((df['l1'] == 0) & (df['l2'] == 0) & (df['l3'] == 0)).sum()
print(f"\n  l1=0 VE l2=0 VE l3=0 (uc faz birlikte sifir): {hepsi_sifir:,} ({hepsi_sifir/len(df)*100:.2f}%)")

# ============================================================
# 3. Gerilim kolonlarindaki eksikliklerin dagilimi
# ============================================================
print("\n" + "=" * 60)
print("3. GERILIM EKSIKLIK DAGILIMI (v1, v2, v3)")
print("=" * 60)

# Birlikte mi eksik?
v_null = df[['v1', 'v2', 'v3']].isnull()
hepsi_eksik = v_null.all(axis=1).sum()
en_az_biri_eksik = v_null.any(axis=1).sum()
hicbiri_eksik_degil = (~v_null.any(axis=1)).sum()

print(f"v1/v2/v3 eksiklik iliskisi:")
print(f"  Uc faz birlikte eksik   : {hepsi_eksik:,}")
print(f"  En az biri eksik        : {en_az_biri_eksik:,}")
print(f"  Hicbiri eksik degil     : {hicbiri_eksik_degil:,}")
print(f"  -> Her zaman birlikte mi: {'EVET' if hepsi_eksik == en_az_biri_eksik else 'HAYIR'}")

# Hangi abonelerde eksik?
print(f"\nAbone bazinda v1 eksiklik:")
abone_v_null = df.groupby('tesisat_no_id').agg(
    toplam=('v1', 'size'),
    eksik=('v1', lambda x: x.isnull().sum())
)
abone_v_null['oran'] = (abone_v_null['eksik'] / abone_v_null['toplam'] * 100).round(1)
abone_v_null = abone_v_null.sort_values('oran', ascending=False)

tam_eksik = abone_v_null[abone_v_null['oran'] == 100]
kismi_eksik = abone_v_null[(abone_v_null['oran'] > 0) & (abone_v_null['oran'] < 100)]
eksik_yok = abone_v_null[abone_v_null['oran'] == 0]

print(f"  %100 eksik olan aboneler : {len(tam_eksik)} abone")
print(f"  Kismen eksik             : {len(kismi_eksik)} abone")
print(f"  Hic eksik olmayan        : {len(eksik_yok)} abone")

if len(tam_eksik) > 0:
    print(f"\n  %100 eksik abonelerin ortak ozellikleri:")
    tam_eksik_ids = tam_eksik.index.tolist()
    tam_eksik_df = df[df['tesisat_no_id'].isin(tam_eksik_ids)]
    print(f"    Marka    : {tam_eksik_df['marka'].value_counts().to_dict()}")
    print(f"    Model    : {tam_eksik_df['model'].value_counts().to_dict()}")
    print(f"    Gerilim  : {tam_eksik_df['gerilim_seviyesi'].value_counts().to_dict()}")

# Dolu voltaj degerlerindeki sifirlar
print(f"\nDolu voltaj degerlerinde sifir kontrolu:")
for col in ['v1', 'v2', 'v3']:
    dolu = df[col].dropna()
    sifir = (dolu == 0).sum()
    negatif = (dolu < 0).sum()
    print(f"  {col}: dolu={len(dolu):,} | sifir={sifir:,} ({sifir/len(dolu)*100:.2f}%) | negatif={negatif:,}")

# ============================================================
# 4. ri ve rc alanlari - eksik ve uc deger kontrolu
# ============================================================
print("\n" + "=" * 60)
print("4. REAKTIF ENERJI (ri, rc) KONTROLU")
print("=" * 60)

for col in ['ri', 'rc']:
    s = df[col]
    dolu = s.dropna()
    print(f"\n  [{col}]")
    print(f"    Null    : {s.isnull().sum():,} ({s.isnull().mean()*100:.2f}%)")
    print(f"    Min     : {dolu.min():.4f}")
    print(f"    Max     : {dolu.max():.4f}")
    print(f"    Mean    : {dolu.mean():.4f}")
    print(f"    Median  : {dolu.median():.4f}")
    print(f"    Sifir   : {(dolu == 0).sum():,} ({(dolu == 0).mean()*100:.2f}%)")
    print(f"    Negatif : {(dolu < 0).sum():,}")
    
    # IQR uc deger
    Q1 = dolu.quantile(0.25)
    Q3 = dolu.quantile(0.75)
    IQR = Q3 - Q1
    ust_sinir = Q3 + 1.5 * IQR
    print(f"    Q1={Q1:.2f} | Q3={Q3:.2f} | IQR={IQR:.2f} | Ust sinir={ust_sinir:.2f}")
    print(f"    > Ust sinir: {(dolu > ust_sinir).sum():,} ({(dolu > ust_sinir).mean()*100:.2f}%)")

# ri/rc birlikte mi eksik?
ri_rc_null = df[['ri', 'rc']].isnull()
print(f"\nri/rc eksiklik iliskisi:")
print(f"  Ikisi birlikte eksik : {ri_rc_null.all(axis=1).sum():,}")
print(f"  En az biri eksik     : {ri_rc_null.any(axis=1).sum():,}")
print(f"  -> Birlikte mi       : {'EVET' if ri_rc_null.all(axis=1).sum() == ri_rc_null.any(axis=1).sum() else 'HAYIR'}")

# Hangi abonelerde eksik?
print(f"\nAbone bazinda ri eksiklik:")
abone_ri_null = df.groupby('tesisat_no_id').agg(
    toplam=('ri', 'size'),
    eksik=('ri', lambda x: x.isnull().sum())
)
abone_ri_null['oran'] = (abone_ri_null['eksik'] / abone_ri_null['toplam'] * 100).round(1)

tam_ri = abone_ri_null[abone_ri_null['oran'] == 100]
if len(tam_ri) > 0:
    print(f"  %100 eksik olan aboneler: {len(tam_ri)} abone")
    tam_ri_ids = tam_ri.index.tolist()
    tam_ri_df = df[df['tesisat_no_id'].isin(tam_ri_ids)]
    print(f"    Marka : {tam_ri_df['marka'].value_counts().to_dict()}")
    print(f"    Model : {tam_ri_df['model'].value_counts().to_dict()}")
else:
    print(f"  %100 eksik olan abone yok")

kismi_ri = abone_ri_null[(abone_ri_null['oran'] > 0) & (abone_ri_null['oran'] < 100)]
print(f"  Kismen eksik: {len(kismi_ri)} abone")
print(f"  Hic eksik yok: {(abone_ri_null['oran'] == 0).sum()} abone")

# v eksikligi ile ri eksikligi arasinda kesisim var mi?
print(f"\nv1 eksik ile ri eksik arasinda kesisim:")
v_eksik_mask = df['v1'].isnull()
ri_eksik_mask = df['ri'].isnull()
print(f"  Hem v1 hem ri eksik : {(v_eksik_mask & ri_eksik_mask).sum():,}")
print(f"  Sadece v1 eksik     : {(v_eksik_mask & ~ri_eksik_mask).sum():,}")
print(f"  Sadece ri eksik     : {(~v_eksik_mask & ri_eksik_mask).sum():,}")
print(f"  Ikisi de dolu       : {(~v_eksik_mask & ~ri_eksik_mask).sum():,}")

# ============================================================
# 5. TEMIZLIK STRATEJISI ONERISI
# ============================================================
print("\n" + "=" * 60)
print("5. TEMIZLIK STRATEJISI ONERISI")
print("=" * 60)

print("""
KOLON BAZINDA KARAR TABLOSU:
============================================================
Kolon             | Durum                    | Oneri
------------------|--------------------------|------------------
tesisat_no_id     | %0 eksik, sorun yok      | OLDUGU GIBI BIRAK
il                | %0 eksik, sorun yok      | OLDUGU GIBI BIRAK
ilce              | %0 eksik, sorun yok      | OLDUGU GIBI BIRAK
gerilim_seviyesi  | %0 eksik, sorun yok      | OLDUGU GIBI BIRAK
marka             | %0 eksik, sorun yok      | OLDUGU GIBI BIRAK
model             | %0 eksik, sorun yok      | OLDUGU GIBI BIRAK
abone_grubu       | %0 eksik, sorun yok      | OLDUGU GIBI BIRAK
son_carpan_degeri | %0 eksik, sorun yok      | OLDUGU GIBI BIRAK
l1                | %0 eksik, sifir var      | OLDUGU GIBI BIRAK (*)
l2                | %0 eksik, sifir var      | OLDUGU GIBI BIRAK (*)
l3                | %0 eksik, sifir var      | OLDUGU GIBI BIRAK (*)
v1                | %21.31 eksik             | SATIRLARI SILME (**)
v2                | %21.31 eksik             | SATIRLARI SILME (**)
v3                | %21.31 eksik             | SATIRLARI SILME (**)
t0                | %0 eksik, min>0          | OLDUGU GIBI BIRAK
ri                | %2.89 eksik              | SATIRLARI SILME (***)
rc                | %2.89 eksik              | SATIRLARI SILME (***)
load_profile_date | %0 eksik, object tipi    | TIP DONUSUMU (****)

ACIKLAMALAR:
------------------------------------------------------------
(*)   l1/l2/l3: Sifir degerler fiziksel olarak gecerli 
      (yuk yok durumu). Negatif deger yok. Silme/doldurma 
      gereksiz. Ancak voltaj analizlerinde sifir akimli 
      periyotlar filtrelenebilir.

(**)  v1/v2/v3: %100 eksik olan 13 abone var - hepsi ayni 
      model (LUN23.5010). Bu sayac modeli voltaj raporlamiyor.
      Doldurmak anlamsiz (veri uretilmis olur). Voltaj 
      analizlerinde bu satirlari cikart, diger analizlerde 
      (tuketim, akim) tut.

(***) ri/rc: %100 eksik olan 7 abone var - hepsi ayni model 
      (LUN23-TF). Ayni mantik: reaktif analiz yapilirken 
      cikar, diger analizlerde tut.

(****) load_profile_date: object -> datetime64 donusumu 
       gerekli. Milisaniye kismi (.000) tum satirlarda ayni, 
       bilgi tasimaz.

EK NOTLAR:
------------------------------------------------------------
- v1/v2/v3 dolu olan satirlarda sifir voltaj degerleri var
  (v1: 4,408 / v2: 2,945 / v3: 58 adet). Bunlar ayri 
  degerlendirilmeli (kesinti? olcum hatasi?).
- t0 kumulatif bir sayac degeri. Negatif veya sifir degeri 
  yok (min=449.62). Tuketim hesabi icin periyodik fark 
  (delta_t0) hesaplanmali.
- ri/rc de kumulatif deger. Sifirlari olan satirlar var 
  (8,803 adet) - bunlar sayac baslangic noktasi olabilir.
""")
