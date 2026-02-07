"""
ADIM 8: Karar Destek Pivot Analizleri
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

# Yardimci kolonlar
df['anomali_var'] = df['Toplam_Anomali_Sayisi'] > 0
df['tarih'] = df['load_profile_date'].dt.date

print(f"Veri: {df.shape[0]:,} satir | Anomalili: {df['anomali_var'].sum():,} ({df['anomali_var'].mean()*100:.1f}%)")

# ============================================================
# 1. IL BAZINDA ANOMALI SAYILARI
# ============================================================
print("\n" + "=" * 65)
print("1. IL BAZINDA ANOMALI SAYILARI")
print("=" * 65)

il_pivot = df.groupby('il').agg(
    toplam_kayit=('anomali_var', 'size'),
    anomalili_kayit=('anomali_var', 'sum'),
    abone_sayisi=('tesisat_no_id', 'nunique'),
    ort_anomali=('Toplam_Anomali_Sayisi', 'mean')
)
il_pivot['anomali_oran'] = (il_pivot['anomalili_kayit'] / il_pivot['toplam_kayit'] * 100).round(1)
il_pivot = il_pivot.sort_values('anomali_oran', ascending=False)

print(f"\n  {'Il':<12} {'Kayit':>10} {'Anomalili':>10} {'Oran%':>8} {'Abone':>6} {'Ort':>6}")
print(f"  {'-'*54}")
for il, row in il_pivot.iterrows():
    print(f"  {str(il):<12} {int(row['toplam_kayit']):>10,} {int(row['anomalili_kayit']):>10,} "
          f"{row['anomali_oran']:>7.1f}% {int(row['abone_sayisi']):>6} {row['ort_anomali']:>6.2f}")

# Il bazinda anomali tipi kirilimi
print(f"\n  Il bazinda anomali tipi dagilimi (kayit sayisi):")
il_tip = df.groupby('il')[anomali_kolonlari].sum().astype(int)
header = f"  {'Il':<12}"
for col in anomali_kolonlari:
    header += f" {col.split('_')[0]:>6}"
print(header)
print(f"  {'-'*len(header)}")
for il, row in il_tip.iterrows():
    line = f"  {str(il):<12}"
    for col in anomali_kolonlari:
        line += f" {row[col]:>6}"
    print(line)

print(f"""
  YORUM: Samsun en yuksek anomali oranina sahip (%51.3). Bunun
  temel sebebi A4 (gerilim eksik) ve A8 (yuksek reaktif) yogunlugu.
  CIKARIM: Samsun bolgesindeki LUNA sayaclar ve reaktif kompanzasyon
  sistemleri oncelikli inceleme gerektirir.
""")

# ============================================================
# 2. ILCE BAZINDA ANOMALI YOGUNLUGU
# ============================================================
print("=" * 65)
print("2. ILCE BAZINDA ANOMALI YOGUNLUGU")
print("=" * 65)

ilce_pivot = df.groupby(['il', 'ilce']).agg(
    toplam_kayit=('anomali_var', 'size'),
    anomalili_kayit=('anomali_var', 'sum'),
    abone_sayisi=('tesisat_no_id', 'nunique'),
    ort_anomali=('Toplam_Anomali_Sayisi', 'mean')
)
ilce_pivot['anomali_oran'] = (ilce_pivot['anomalili_kayit'] / ilce_pivot['toplam_kayit'] * 100).round(1)
ilce_pivot = ilce_pivot.sort_values('anomali_oran', ascending=False)

print(f"\n  {'Il/Ilce':<28} {'Kayit':>8} {'Anomalili':>10} {'Oran%':>8} {'Abone':>6}")
print(f"  {'-'*62}")
for (il, ilce), row in ilce_pivot.iterrows():
    label = f"{il}/{ilce}"
    if len(label) > 27:
        label = label[:27]
    print(f"  {label:<28} {int(row['toplam_kayit']):>8,} {int(row['anomalili_kayit']):>10,} "
          f"{row['anomali_oran']:>7.1f}% {int(row['abone_sayisi']):>6}")

print(f"""
  YORUM: En yuksek anomali oranli ilceler Kavak, Terme, Atakum,
  Vezirkopru (Samsun) ve Hamamozu, Tasova (Amasya). Bu ilcelerde
  %98-100 anomali orani var.
  CIKARIM: Bu ilcelerdeki aboneler buyuk olasilikla tek sayac
  modeline (LUN23.5010) bagli. Saha ekipleri bu bolgelere
  yonlendirilmeli.
""")

# ============================================================
# 3. ABONE GRUBU BAZINDA RISK DAGILIMI
# ============================================================
print("=" * 65)
print("3. ABONE GRUBU BAZINDA RISK DAGILIMI")
print("=" * 65)

ag_pivot = df.groupby('abone_grubu').agg(
    toplam_kayit=('anomali_var', 'size'),
    anomalili_kayit=('anomali_var', 'sum'),
    abone_sayisi=('tesisat_no_id', 'nunique'),
)
ag_pivot['anomali_oran'] = (ag_pivot['anomalili_kayit'] / ag_pivot['toplam_kayit'] * 100).round(1)
ag_pivot = ag_pivot.sort_values('anomali_oran', ascending=False)

print(f"\n  {'Abone Grubu':<38} {'Kayit':>8} {'Oran%':>8} {'Abone':>6}")
print(f"  {'-'*62}")
for ag, row in ag_pivot.iterrows():
    label = str(ag)[:37]
    print(f"  {label:<38} {int(row['toplam_kayit']):>8,} {row['anomali_oran']:>7.1f}% {int(row['abone_sayisi']):>6}")

# Abone grubu x anomali tipi
print(f"\n  Abone grubu x anomali tipi (% oran, satir bazinda):")
ag_tip_pct = df.groupby('abone_grubu')[anomali_kolonlari].mean() * 100
header = f"  {'Grup':<25}"
for col in anomali_kolonlari:
    header += f" {col.split('_')[0]:>6}"
print(header)
for ag, row in ag_tip_pct.iterrows():
    label = str(ag)[:24]
    line = f"  {label:<25}"
    for col in anomali_kolonlari:
        line += f" {row[col]:>5.1f}%"
    print(line)

print(f"""
  YORUM: "Tek Terimli Aydinlatma AG" (%100) ve "Tek Terimli 
  Tarimsal Sulama AG" (%96.2) en yuksek risk gruplari.
  Ticarethane AG de %46.2 ile dikkat cekiyor.
  CIKARIM: AG abone gruplarinda risk belirgin sekilde yuksek.
  Aydinlatma ve sulama aboneleri icin sayac donanimlarinin
  gozden gecirilmesi gerekiyor.
""")

# ============================================================
# 4. SAYAC MARKA / MODEL BAZINDA RISK PROFILI
# ============================================================
print("=" * 65)
print("4. SAYAC MARKA / MODEL BAZINDA RISK PROFILI")
print("=" * 65)

# Marka bazinda
print("\n  --- Marka Bazinda ---")
marka_pivot = df.groupby('marka').agg(
    toplam_kayit=('anomali_var', 'size'),
    anomalili_kayit=('anomali_var', 'sum'),
    abone_sayisi=('tesisat_no_id', 'nunique'),
)
marka_pivot['anomali_oran'] = (marka_pivot['anomalili_kayit'] / marka_pivot['toplam_kayit'] * 100).round(1)

print(f"  {'Marka':<10} {'Kayit':>10} {'Anomalili':>10} {'Oran%':>8} {'Abone':>6}")
print(f"  {'-'*46}")
for marka, row in marka_pivot.iterrows():
    print(f"  {str(marka):<10} {int(row['toplam_kayit']):>10,} {int(row['anomalili_kayit']):>10,} "
          f"{row['anomali_oran']:>7.1f}% {int(row['abone_sayisi']):>6}")

# Model bazinda
print(f"\n  --- Model Bazinda ---")
model_pivot = df.groupby(['marka', 'model']).agg(
    toplam_kayit=('anomali_var', 'size'),
    anomalili_kayit=('anomali_var', 'sum'),
    abone_sayisi=('tesisat_no_id', 'nunique'),
)
model_pivot['anomali_oran'] = (model_pivot['anomalili_kayit'] / model_pivot['toplam_kayit'] * 100).round(1)
model_pivot = model_pivot.sort_values('anomali_oran', ascending=False)

print(f"  {'Marka/Model':<28} {'Kayit':>10} {'Anomalili':>10} {'Oran%':>8} {'Abone':>6}")
print(f"  {'-'*64}")
for (marka, model), row in model_pivot.iterrows():
    label = f"{marka}/{model}"
    print(f"  {label:<28} {int(row['toplam_kayit']):>10,} {int(row['anomalili_kayit']):>10,} "
          f"{row['anomali_oran']:>7.1f}% {int(row['abone_sayisi']):>6}")

# Model x anomali tipi
print(f"\n  Model bazinda anomali tipi (kayit sayisi):")
model_tip = df.groupby('model')[anomali_kolonlari].sum().astype(int)
model_tip = model_tip.loc[model_pivot.reset_index()['model'].values]

header = f"  {'Model':<18}"
for col in anomali_kolonlari:
    header += f" {col.split('_')[0]:>6}"
print(header)
for model, row in model_tip.iterrows():
    line = f"  {str(model):<18}"
    for col in anomali_kolonlari:
        line += f" {row[col]:>6}"
    print(line)

print(f"""
  YORUM: LUNA marka %98.6 anomali oranina sahip. Model bazinda
  LUN23.5010 (%100) ve LUN23-TF (%91.2) en riskli. MAKEL
  modelleri %18.6 ile cok daha dusuk.
  CIKARIM: LUNA sayaclarinda A4 (gerilim eksik) yapisal sorundur.
  LUN23-TF'de ek olarak A5 (sifir tuketim) ve A8 (reaktif)
  anomalileri belirgin. LUNA sayaclarinin yenilenmesi veya
  firmware guncellenmesi onerilebilir.
""")

# ============================================================
# 5. GUN BAZLI ANOMALI FREKANSI
# ============================================================
print("=" * 65)
print("5. GUN BAZLI ANOMALI FREKANSI")
print("=" * 65)

gun_pivot = df.groupby('tarih').agg(
    toplam_kayit=('anomali_var', 'size'),
    anomalili_kayit=('anomali_var', 'sum'),
)
gun_pivot['anomali_oran'] = (gun_pivot['anomalili_kayit'] / gun_pivot['toplam_kayit'] * 100).round(1)

print(f"\n  Genel istatistikler:")
print(f"    Toplam gun       : {len(gun_pivot)}")
print(f"    Ort anomali/gun  : {gun_pivot['anomalili_kayit'].mean():.0f} kayit")
print(f"    Ort anomali oran : %{gun_pivot['anomali_oran'].mean():.1f}")
print(f"    Min anomali oran : %{gun_pivot['anomali_oran'].min():.1f} ({gun_pivot['anomali_oran'].idxmin()})")
print(f"    Max anomali oran : %{gun_pivot['anomali_oran'].max():.1f} ({gun_pivot['anomali_oran'].idxmax()})")

# En yuksek ve en dusuk 5 gun
print(f"\n  En yuksek anomali oranli 5 gun:")
for tarih, row in gun_pivot.nlargest(5, 'anomali_oran').iterrows():
    gun_adi = pd.Timestamp(tarih).day_name()
    print(f"    {tarih} ({gun_adi[:3]}) | {int(row['anomalili_kayit']):>5} / {int(row['toplam_kayit']):>5} | %{row['anomali_oran']:.1f}")

print(f"\n  En dusuk anomali oranli 5 gun:")
for tarih, row in gun_pivot.nsmallest(5, 'anomali_oran').iterrows():
    gun_adi = pd.Timestamp(tarih).day_name()
    print(f"    {tarih} ({gun_adi[:3]}) | {int(row['anomalili_kayit']):>5} / {int(row['toplam_kayit']):>5} | %{row['anomali_oran']:.1f}")

# Hafta ici / hafta sonu
df['gun_tipi'] = np.where(df['load_profile_date'].dt.dayofweek < 5, 'Hafta Ici', 'Hafta Sonu')
gun_tipi_pivot = df.groupby('gun_tipi').agg(
    toplam=('anomali_var', 'size'),
    anomalili=('anomali_var', 'sum'),
)
gun_tipi_pivot['oran'] = (gun_tipi_pivot['anomalili'] / gun_tipi_pivot['toplam'] * 100).round(1)

print(f"\n  Hafta ici / Hafta sonu:")
for tip, row in gun_tipi_pivot.iterrows():
    print(f"    {tip:<12}: {int(row['anomalili']):>8,} / {int(row['toplam']):>8,} | %{row['oran']:.1f}")

# Haftanin gunu bazinda
gun_adi_pivot = df.groupby(df['load_profile_date'].dt.day_name()).agg(
    toplam=('anomali_var', 'size'),
    anomalili=('anomali_var', 'sum'),
)
gun_adi_pivot['oran'] = (gun_adi_pivot['anomalili'] / gun_adi_pivot['toplam'] * 100).round(1)
gun_adi_pivot = gun_adi_pivot.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

print(f"\n  Haftanin gunu bazinda:")
for gun, row in gun_adi_pivot.iterrows():
    print(f"    {gun:<12}: %{row['oran']:.1f}")

print(f"""
  YORUM: Anomali orani gunler arasinda oldukca sabit (%37-38
  araligi). Hafta ici ve hafta sonu arasinda anlamli fark yok.
  CIKARIM: Anomaliler zamansal degil, yapisal kaynakli (sayac
  modeli, donanim sorunu). Gunun veya haftanin belirli bir
  zamaninda yogunlasma gorulmuyor.
""")

# ============================================================
# 6. EN COK ANOMALI URETEN ILK 10 TESISAT
# ============================================================
print("=" * 65)
print("6. EN COK ANOMALI URETEN ILK 10 TESISAT")
print("=" * 65)

tesisat_pivot = df.groupby('tesisat_no_id').agg(
    toplam_kayit=('anomali_var', 'size'),
    anomalili_kayit=('anomali_var', 'sum'),
    toplam_anomali=('Toplam_Anomali_Sayisi', 'sum'),
    il=('il', 'first'),
    ilce=('ilce', 'first'),
    gerilim=('gerilim_seviyesi', 'first'),
    model=('model', 'first'),
    abone_grubu=('abone_grubu', 'first'),
)
tesisat_pivot['anomali_oran'] = (tesisat_pivot['anomalili_kayit'] / tesisat_pivot['toplam_kayit'] * 100).round(1)
tesisat_pivot = tesisat_pivot.sort_values('toplam_anomali', ascending=False)

print(f"\n  {'#':<3} {'Tesisat':<20} {'Il/Ilce':<20} {'Model':<16} {'Kayit':>6} {'Anomali':>8} {'Oran':>7} {'Toplam':>7}")
print(f"  {'-'*85}")

for i, (tesisat, row) in enumerate(tesisat_pivot.head(10).iterrows(), 1):
    ilce_str = f"{row['il']}/{row['ilce']}"
    if len(ilce_str) > 19:
        ilce_str = ilce_str[:19]
    print(f"  {i:<3} {tesisat[:18]}.. {ilce_str:<20} {str(row['model']):<16} "
          f"{int(row['toplam_kayit']):>6} {int(row['anomalili_kayit']):>8} "
          f"{row['anomali_oran']:>6.1f}% {int(row['toplam_anomali']):>7}")

# Her tesisat icin anomali tipi kirilimi
print(f"\n  Ilk 10 tesisat - anomali tipi kirilimi:")
print(f"  {'Tesisat':<20}", end="")
for col in anomali_kolonlari:
    print(f" {col.split('_')[0]:>5}", end="")
print()

for tesisat, _ in tesisat_pivot.head(10).iterrows():
    sub = df[df['tesisat_no_id'] == tesisat]
    print(f"  {tesisat[:18]}..", end="")
    for col in anomali_kolonlari:
        sayi = sub[col].sum()
        print(f" {sayi:>5}", end="")
    print()

print(f"""
  YORUM: Ilk 10 tesisat 8,000-11,700 arasi toplam anomali puanina
  sahip. 23bfc4eed916 tamamiyla A5 (sifir tuketim) -- sayac 2 aydir
  kayit uretmiyor. 21d7911f836e hem A4 (gerilim eksik) hem A8
  (reaktif) hem A7 (gece anomali) birlesimi gosteriyor.
  CIKARIM: Bu 10 tesisata oncelikli saha ziyareti planlanmalidir.
  - Sifir tuketimli tesisatlar: sayac arizasi kontrolu
  - LUNA tesisatlar: sayac degisimi/firmware
  - Yuksek reaktif: kompanzasyon panosu kontrolu
""")

# Temizle
df = df.drop(columns=['anomali_var', 'tarih', 'gun_tipi'], errors='ignore')
