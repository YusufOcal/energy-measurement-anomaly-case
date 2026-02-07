"""
ADIM 6: Anomali Kurallarini Tanimla ve Uygula
Temiz veri + turetilmis degiskenler uzerinden (dataset_clean.csv).
Henuz veri degistirmiyoruz, sadece tespit + raporlama.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("dataset_clean.csv", index_col=0)
df['load_profile_date'] = pd.to_datetime(df['load_profile_date'])

print(f"Veri: {df.shape[0]:,} satir, {df.shape[1]} kolon")

# ============================================================
# ANOMALI 1: Akim var, tuketim yok
# ============================================================
print("\n" + "=" * 65)
print("ANOMALI 1: AKIM VAR, TUKETIM YOK")
print("=" * 65)
print("""
  Kural  : Ortalama_Akim > 1A  VE  Aktif_Tuketim_Farki == 0
  Risk   : Sayac akim cekiyor ama tuketim kaydetmiyor.
           Sayac arizasi, bypass veya kayit hatasi olabilir.
  Anlam  : Enerji cekilmesine ragmen faturalandirilmamis
           tuketim riski (kacak veya sayac donmasi).
""")

mask_1 = (df['Ortalama_Akim'] > 1) & (df['Aktif_Tuketim_Farki'] == 0)
anomali_1 = df[mask_1]

print(f"  Tespit edilen kayit : {len(anomali_1):,}")
print(f"  Etkilenen abone     : {anomali_1['tesisat_no_id'].nunique()}")

if len(anomali_1) > 0:
    print(f"\n  Abone bazinda dagilim:")
    a1_per_abone = anomali_1.groupby('tesisat_no_id').agg(
        sayi=('Aktif_Tuketim_Farki', 'size'),
        ort_akim=('Ortalama_Akim', 'mean')
    ).sort_values('sayi', ascending=False)
    for idx, row in a1_per_abone.iterrows():
        toplam = len(df[df['tesisat_no_id'] == idx])
        print(f"    {idx[:16]}... | {int(row['sayi'])} kayit ({row['sayi']/toplam*100:.1f}%) | ort akim: {row['ort_akim']:.2f}A")

    print(f"\n  Ornek satirlar:")
    orn = anomali_1[['load_profile_date', 'tesisat_no_id', 'l1', 'l2', 'l3', 'Ortalama_Akim', 'Aktif_Tuketim_Farki']].head(5)
    for _, row in orn.iterrows():
        print(f"    {row['load_profile_date']} | {row['tesisat_no_id'][:12]}... | "
              f"l1={row['l1']:.2f} l2={row['l2']:.2f} l3={row['l3']:.2f} | "
              f"OrtAkim={row['Ortalama_Akim']:.2f}A | delta_t0={row['Aktif_Tuketim_Farki']:.4f}")

# ============================================================
# ANOMALI 2: Tuketim var, akim dusuk
# ============================================================
print("\n" + "=" * 65)
print("ANOMALI 2: TUKETIM VAR, AKIM DUSUK")
print("=" * 65)
print("""
  Kural  : Aktif_Tuketim_Farki > 1 kWh  VE  Ortalama_Akim < 0.5A
  Risk   : Sayac tuketim kaydediyor ama fiziksel akim cok dusuk.
           Olcum hatasi, sayac kayma sorunu veya veri tutarsizligi.
  Anlam  : Musteriye fazla fatura kesilme riski veya
           sayac kalibrasyon sorunu.
""")

mask_2 = (df['Aktif_Tuketim_Farki'] > 1) & (df['Ortalama_Akim'] < 0.5)
anomali_2 = df[mask_2]

print(f"  Tespit edilen kayit : {len(anomali_2):,}")
print(f"  Etkilenen abone     : {anomali_2['tesisat_no_id'].nunique()}")

if len(anomali_2) > 0:
    a2_per_abone = anomali_2.groupby('tesisat_no_id').agg(
        sayi=('Aktif_Tuketim_Farki', 'size'),
        ort_tuketim=('Aktif_Tuketim_Farki', 'mean'),
        ort_akim=('Ortalama_Akim', 'mean')
    ).sort_values('sayi', ascending=False)
    print(f"\n  Abone bazinda dagilim (ilk 10):")
    for idx, row in a2_per_abone.head(10).iterrows():
        toplam = len(df[df['tesisat_no_id'] == idx])
        gs = df[df['tesisat_no_id'] == idx]['gerilim_seviyesi'].iloc[0]
        print(f"    {idx[:16]}... | {int(row['sayi'])} kayit ({row['sayi']/toplam*100:.1f}%) | "
              f"ort tuketim: {row['ort_tuketim']:.2f}kWh | ort akim: {row['ort_akim']:.2f}A | {gs}")

    print(f"\n  Ornek satirlar:")
    orn = anomali_2[['load_profile_date', 'tesisat_no_id', 'Ortalama_Akim', 'Aktif_Tuketim_Farki']].head(5)
    for _, row in orn.iterrows():
        print(f"    {row['load_profile_date']} | {row['tesisat_no_id'][:12]}... | "
              f"OrtAkim={row['Ortalama_Akim']:.2f}A | delta_t0={row['Aktif_Tuketim_Farki']:.4f}kWh")

# ============================================================
# ANOMALI 3: Faz dengesizligi + yuksek akim
# ============================================================
print("\n" + "=" * 65)
print("ANOMALI 3: FAZ DENGESIZLIGI + YUKSEK AKIM")
print("=" * 65)
print("""
  Kural  : Faz_Dengesizligi > 30A  VE  Ortalama_Akim > 10A
  Risk   : Yuksek akim altinda faz yukleri cok dengesiz.
           Trafo, kablo asiri isinmasi; notr hattinda asiri yuk.
  Anlam  : Ekipman hasari ve enerji kaybi riski.
           Teknik mudehaley (yuk dengeleme) gerektirir.
""")

mask_3 = (df['Faz_Dengesizligi'] > 30) & (df['Ortalama_Akim'] > 10)
anomali_3 = df[mask_3]

print(f"  Tespit edilen kayit : {len(anomali_3):,}")
print(f"  Etkilenen abone     : {anomali_3['tesisat_no_id'].nunique()}")

if len(anomali_3) > 0:
    a3_per_abone = anomali_3.groupby('tesisat_no_id').agg(
        sayi=('Faz_Dengesizligi', 'size'),
        ort_dengesizlik=('Faz_Dengesizligi', 'mean'),
        max_dengesizlik=('Faz_Dengesizligi', 'max'),
        ort_akim=('Ortalama_Akim', 'mean')
    ).sort_values('sayi', ascending=False)
    print(f"\n  Abone bazinda dagilim:")
    for idx, row in a3_per_abone.iterrows():
        toplam = len(df[df['tesisat_no_id'] == idx])
        gs = df[df['tesisat_no_id'] == idx]['gerilim_seviyesi'].iloc[0]
        print(f"    {idx[:16]}... | {int(row['sayi'])} kayit ({row['sayi']/toplam*100:.1f}%) | "
              f"ort deng: {row['ort_dengesizlik']:.1f}A | max deng: {row['max_dengesizlik']:.1f}A | {gs}")

# ============================================================
# ANOMALI 4: Gerilim verisi eksik, tuketim devam ediyor
# ============================================================
print("\n" + "=" * 65)
print("ANOMALI 4: GERILIM EKSIK, TUKETIM DEVAM EDIYOR")
print("=" * 65)
print("""
  Kural  : Ortalama_Gerilim == NaN  VE  Aktif_Tuketim_Farki > 0
  Risk   : Sayac enerji tuketimi kaydediyor ama gerilim olcemiyor.
           Gerilim sensoru arizali olabilir veya sayac modeli
           gerilim desteklemiyor.
  Anlam  : Gerilim tabanli anomali tespiti bu abonelerde yapilamaz.
           Veri kalitesi ve dogrulama eksikligi.
""")

mask_4 = (df['Ortalama_Gerilim'].isna()) & (df['Aktif_Tuketim_Farki'] > 0)
anomali_4 = df[mask_4]

print(f"  Tespit edilen kayit : {len(anomali_4):,}")
print(f"  Etkilenen abone     : {anomali_4['tesisat_no_id'].nunique()}")

if len(anomali_4) > 0:
    a4_models = anomali_4.groupby('model').size().sort_values(ascending=False)
    print(f"\n  Model bazinda dagilim:")
    for model, sayi in a4_models.items():
        print(f"    {model}: {sayi:,} kayit")

    a4_per_abone = anomali_4.groupby('tesisat_no_id').agg(
        sayi=('Aktif_Tuketim_Farki', 'size'),
        ort_tuketim=('Aktif_Tuketim_Farki', 'mean')
    ).sort_values('sayi', ascending=False)
    print(f"\n  Abone bazinda (ilk 5):")
    for idx, row in a4_per_abone.head(5).iterrows():
        model = df[df['tesisat_no_id'] == idx]['model'].iloc[0]
        print(f"    {idx[:16]}... | {int(row['sayi'])} kayit | ort tuketim: {row['ort_tuketim']:.4f}kWh | {model}")

# ============================================================
# ANOMALI 5: Negatif veya sifir Aktif_Tuketim_Farki
# ============================================================
print("\n" + "=" * 65)
print("ANOMALI 5: NEGATIF VEYA SIFIR AKTIF_TUKETIM_FARKI")
print("=" * 65)
print("""
  Kural  : Aktif_Tuketim_Farki <= 0  (NaN haric)
  Risk   : Negatif -> sayac geri sarmis (manipulasyon?).
           Sifir   -> uzun sure hic tuketim yok (sayac durmus
           veya gercekten yuk yok).
  Anlam  : Negatif degerler acil inceleme gerektirir.
           Sifir degerler tekrar ediyorsa sayac arizasi olabilir.
""")

negatif_mask = (df['Aktif_Tuketim_Farki'] < 0)
sifir_mask = (df['Aktif_Tuketim_Farki'] == 0)

negatif_count = negatif_mask.sum()
sifir_count = sifir_mask.sum()

print(f"  Negatif (< 0)  : {negatif_count:,}")
print(f"  Sifir   (== 0) : {sifir_count:,} ({sifir_count/len(df)*100:.2f}%)")

if sifir_count > 0:
    a5_per_abone = df[sifir_mask].groupby('tesisat_no_id').size().sort_values(ascending=False)
    print(f"\n  Sifir tuketim - abone bazinda (ilk 10):")
    for idx, sayi in a5_per_abone.head(10).items():
        toplam = len(df[df['tesisat_no_id'] == idx])
        gs = df[df['tesisat_no_id'] == idx]['gerilim_seviyesi'].iloc[0]
        print(f"    {idx[:16]}... | {sayi} kayit ({sayi/toplam*100:.1f}%) | {gs}")

# Ardisik sifirlar (uzun kesinti belirtisi)
print(f"\n  Ardisik sifir tuketim analizi:")
df['sifir_tuketim'] = (df['Aktif_Tuketim_Farki'] == 0).astype(int)
df['sifir_grup'] = (df['sifir_tuketim'] != df.groupby('tesisat_no_id')['sifir_tuketim'].shift()).cumsum()
ardisik = df[df['sifir_tuketim'] == 1].groupby(['tesisat_no_id', 'sifir_grup']).size()
if len(ardisik) > 0:
    print(f"    Ardisik sifir blok sayisi: {len(ardisik)}")
    print(f"    Ort blok uzunlugu: {ardisik.mean():.1f} kayit")
    print(f"    Max blok uzunlugu: {ardisik.max()} kayit")
    uzun_blok = ardisik[ardisik >= 10]
    print(f"    10+ ardisik sifir: {len(uzun_blok)} blok")
df = df.drop(columns=['sifir_tuketim', 'sifir_grup'])

# ============================================================
# ANOMALI 6: Uzun sure sabit tuketim
# ============================================================
print("\n" + "=" * 65)
print("ANOMALI 6: UZUN SURE SABIT TUKETIM")
print("=" * 65)
print("""
  Kural  : Ardisik >=8 kayit ayni Aktif_Tuketim_Farki degerine
           sahip (>0 ve NaN olmayan).
  Risk   : Dogal tuketim dalgalanma gosterir. Uzun sure tam
           sabit tuketim sayac arizasi veya veri kopyalama
           isareti olabilir.
  Anlam  : Sayacin gercek olcum yapip yapmadiginin sorgulanmasi.
""")

sabit_esik = 8  # ardisik kayit
anomali_6_toplam = 0
anomali_6_aboneler = []

for tesisat in df['tesisat_no_id'].unique():
    sub = df[df['tesisat_no_id'] == tesisat]['Aktif_Tuketim_Farki'].dropna()
    # Pozitif tuketimler
    sub_pos = sub[sub > 0]
    if len(sub_pos) < sabit_esik:
        continue
    
    # Ardisik ayni degerleri bul
    fark = sub_pos.diff().ne(0).cumsum()
    gruplar = sub_pos.groupby(fark).agg(['count', 'first'])
    sabit_bloklar = gruplar[gruplar['count'] >= sabit_esik]
    
    if len(sabit_bloklar) > 0:
        for _, blok in sabit_bloklar.iterrows():
            anomali_6_toplam += int(blok['count'])
            anomali_6_aboneler.append({
                'tesisat': tesisat[:16],
                'uzunluk': int(blok['count']),
                'deger': blok['first']
            })

print(f"  Tespit edilen toplam kayit : {anomali_6_toplam}")
print(f"  Sabit blok sayisi          : {len(anomali_6_aboneler)}")

if len(anomali_6_aboneler) > 0:
    a6_df = pd.DataFrame(anomali_6_aboneler).sort_values('uzunluk', ascending=False)
    print(f"  Etkilenen abone            : {a6_df['tesisat'].nunique()}")
    print(f"\n  Sabit bloklar (ilk 15):")
    for _, row in a6_df.head(15).iterrows():
        print(f"    {row['tesisat']}... | {row['uzunluk']} ardisik | sabit deger: {row['deger']:.4f} kWh")

# ============================================================
# ANOMALI 7: Gece saatlerinde olagandisi tuketim
# ============================================================
print("\n" + "=" * 65)
print("ANOMALI 7: GECE SAATLERINDE OLAGANDISI TUKETIM")
print("=" * 65)
print("""
  Kural  : Saat_Dilimi == 'Gece' VE Aktif_Tuketim_Farki > 
           (abonenin kendi gece ortalamasinin 3 kati)
  Risk   : Gece saatlerinde (22-06) beklenmeyen yuksek tuketim.
           Izinsiz kullanim veya sayac manipulasyonu olabilir.
  Anlam  : Normal is saatleri disinda anormal enerji cekisi.
""")

# Her abone icin gece ortalamasini hesapla
gece_mask = df['Saat_Dilimi'] == 'Gece'
gece_tuketim = df[gece_mask & (df['Aktif_Tuketim_Farki'] > 0)]

abone_gece_ort = gece_tuketim.groupby('tesisat_no_id')['Aktif_Tuketim_Farki'].agg(['mean', 'std'])
abone_gece_ort.columns = ['gece_ort', 'gece_std']

# Esik: kendi gece ortalamasinin 3 kati
anomali_7_list = []
for tesisat in df['tesisat_no_id'].unique():
    if tesisat not in abone_gece_ort.index:
        continue
    gece_ort = abone_gece_ort.loc[tesisat, 'gece_ort']
    if gece_ort <= 0:
        continue
    esik = gece_ort * 3
    
    sub = df[(df['tesisat_no_id'] == tesisat) & gece_mask & (df['Aktif_Tuketim_Farki'] > esik)]
    if len(sub) > 0:
        anomali_7_list.append({
            'tesisat': tesisat[:16],
            'sayi': len(sub),
            'gece_ort': gece_ort,
            'esik': esik,
            'max_tuketim': sub['Aktif_Tuketim_Farki'].max()
        })

a7_df = pd.DataFrame(anomali_7_list).sort_values('sayi', ascending=False) if anomali_7_list else pd.DataFrame()

total_a7 = a7_df['sayi'].sum() if len(a7_df) > 0 else 0
print(f"  Tespit edilen kayit : {total_a7:,}")
print(f"  Etkilenen abone     : {len(a7_df)}")

if len(a7_df) > 0:
    print(f"\n  Abone bazinda (ilk 10):")
    for _, row in a7_df.head(10).iterrows():
        print(f"    {row['tesisat']}... | {int(row['sayi'])} kayit | "
              f"gece ort: {row['gece_ort']:.4f} | esik: {row['esik']:.4f} | "
              f"max: {row['max_tuketim']:.4f} kWh")

# ============================================================
# ANOMALI 8: Yuksek reaktif + dusuk aktif tuketim
# ============================================================
print("\n" + "=" * 65)
print("ANOMALI 8: YUKSEK REAKTIF + DUSUK AKTIF TUKETIM")
print("=" * 65)
print("""
  Kural  : delta_ri > 0 VE delta_ri / Aktif_Tuketim_Farki > 0.62
           (cos_phi < 0.85 esigine karsilik gelir)
           Sadece ri/rc dolu ve Aktif_Tuketim_Farki > 0 olanlar.
  Risk   : Yuksek reaktif tuketim, dusuk guc faktoru.
           Sebeke kalitesini bozar, kayiplari arttirir.
  Anlam  : Aboneye reaktif ceza uygulanmasi gerektigi noktalar.
           Kompanzasyon sistemi eksik veya arizali.
""")

# delta_ri hesapla (ri kumulatif, tesisat bazinda fark)
df['delta_ri'] = df.groupby('tesisat_no_id')['ri'].diff()
df['delta_rc'] = df.groupby('tesisat_no_id')['rc'].diff()

# Reaktif / Aktif oran
valid_mask = (
    df['delta_ri'].notna() & 
    (df['delta_ri'] > 0) & 
    (df['Aktif_Tuketim_Farki'] > 0)
)
valid = df[valid_mask].copy()
valid['ri_t0_oran'] = valid['delta_ri'] / valid['Aktif_Tuketim_Farki']

# cos_phi < 0.85 -> tan_phi > 0.62 -> ri/t0 > 0.62
esik_oran = 0.62
mask_8 = valid['ri_t0_oran'] > esik_oran
anomali_8 = valid[mask_8]

print(f"  Analiz edilen kayit       : {len(valid):,} (ri dolu, pozitif tuketim)")
print(f"  Tespit edilen kayit       : {len(anomali_8):,} ({len(anomali_8)/len(valid)*100:.1f}%)")
print(f"  Etkilenen abone           : {anomali_8['tesisat_no_id'].nunique()}")

if len(anomali_8) > 0:
    a8_per_abone = anomali_8.groupby('tesisat_no_id').agg(
        sayi=('ri_t0_oran', 'size'),
        ort_oran=('ri_t0_oran', 'mean')
    ).sort_values('sayi', ascending=False)
    print(f"\n  Abone bazinda (ilk 10):")
    for idx, row in a8_per_abone.head(10).iterrows():
        toplam_valid = len(valid[valid['tesisat_no_id'] == idx])
        gs = df[df['tesisat_no_id'] == idx]['gerilim_seviyesi'].iloc[0]
        print(f"    {idx[:16]}... | {int(row['sayi'])} kayit ({row['sayi']/toplam_valid*100:.1f}%) | "
              f"ort ri/t0: {row['ort_oran']:.2f} | {gs}")

# Temizle
df = df.drop(columns=['delta_ri', 'delta_rc'], errors='ignore')

# ============================================================
# GENEL OZET TABLOSU
# ============================================================
print("\n" + "=" * 65)
print("GENEL ANOMALI OZET TABLOSU")
print("=" * 65)

print(f"""
  {'#':<4} {'Anomali':<40} {'Kayit':>8} {'Abone':>6}
  {'-'*62}
  {'1':<4} {'Akim var, tuketim yok':<40} {len(anomali_1):>8,} {anomali_1['tesisat_no_id'].nunique():>6}
  {'2':<4} {'Tuketim var, akim dusuk':<40} {len(anomali_2):>8,} {anomali_2['tesisat_no_id'].nunique():>6}
  {'3':<4} {'Faz dengesizligi + yuksek akim':<40} {len(anomali_3):>8,} {anomali_3['tesisat_no_id'].nunique():>6}
  {'4':<4} {'Gerilim eksik, tuketim var':<40} {len(anomali_4):>8,} {anomali_4['tesisat_no_id'].nunique():>6}
  {'5':<4} {'Negatif/sifir tuketim farki':<40} {negatif_count + sifir_count:>8,} {'--':>6}
  {'6':<4} {'Uzun sure sabit tuketim':<40} {anomali_6_toplam:>8,} {a6_df['tesisat'].nunique() if len(anomali_6_aboneler) > 0 else 0:>6}
  {'7':<4} {'Gece olagandisi tuketim':<40} {total_a7:>8,} {len(a7_df):>6}
  {'8':<4} {'Yuksek reaktif / dusuk aktif':<40} {len(anomali_8):>8,} {anomali_8['tesisat_no_id'].nunique():>6}
""")

print("NOT: Bir kayit birden fazla anomaliye dahil olabilir.")
print("     Veri uzerinde degisiklik YAPILMADI, sadece tespit.")
