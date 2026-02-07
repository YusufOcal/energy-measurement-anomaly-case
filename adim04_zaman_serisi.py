"""
ADIM 4: Zaman Serisi Duzenlemesi
Temiz veri uzerinden calisiyor (dataset_clean.csv).
"""

import pandas as pd
import numpy as np

df = pd.read_csv("dataset_clean.csv", index_col=0)

# ============================================================
# 1. load_profile_date -> datetime donusumu
# ============================================================
print("=" * 60)
print("1. TARIH DONUSUMU")
print("=" * 60)

print(f"Oncesi  : {df['load_profile_date'].dtype}")
df['load_profile_date'] = pd.to_datetime(df['load_profile_date'])
print(f"Sonrasi : {df['load_profile_date'].dtype}")
print(f"Min     : {df['load_profile_date'].min()}")
print(f"Max     : {df['load_profile_date'].max()}")
print(f"Aralik  : {df['load_profile_date'].max() - df['load_profile_date'].min()}")
print(f"Null    : {df['load_profile_date'].isnull().sum()}")

# ============================================================
# 2. Tarih-saat bilesenleri dogrulama (15 dakikalik periyotlar)
# ============================================================
print("\n" + "=" * 60)
print("2. PERIYOT DOGRULAMA (15 DAKIKA)")
print("=" * 60)

# Dakika dagilimi
dakika = df['load_profile_date'].dt.minute
dakika_dist = dakika.value_counts().sort_index()

beklenen_dakikalar = [0, 15, 30, 45]
standart_kayit = dakika.isin(beklenen_dakikalar).sum()
standart_disi = (~dakika.isin(beklenen_dakikalar)).sum()

print(f"Beklenen dakika degerleri (0, 15, 30, 45):")
for d in beklenen_dakikalar:
    sayi = (dakika == d).sum()
    print(f"  :{d:02d} -> {sayi:,} kayit")

print(f"\nStandart (0/15/30/45)  : {standart_kayit:,} ({standart_kayit/len(df)*100:.2f}%)")
print(f"Standart disi          : {standart_disi:,} ({standart_disi/len(df)*100:.2f}%)")

if standart_disi > 0:
    print(f"\nStandart disi dakika degerleri (ilk 20):")
    diger = dakika_dist[~dakika_dist.index.isin(beklenen_dakikalar)]
    for dk, sayi in diger.head(20).items():
        print(f"  :{dk:02d} -> {sayi:,} kayit")

# Saniye kontrolu
saniye = df['load_profile_date'].dt.second
saniye_sifir_olmayan = (saniye != 0).sum()
print(f"\nSaniye != 0 olan kayit : {saniye_sifir_olmayan:,}")

# ============================================================
# 3. Veriyi tesisat_no_id ve zaman bazinda sirala
# ============================================================
print("\n" + "=" * 60)
print("3. SIRALAMA")
print("=" * 60)

print(f"Siralama oncesi (ilk 3 satirin tarihleri):")
print(f"  [{df.index[0]}] {df['load_profile_date'].iloc[0]} | {df['tesisat_no_id'].iloc[0][:12]}...")
print(f"  [{df.index[1]}] {df['load_profile_date'].iloc[1]} | {df['tesisat_no_id'].iloc[1][:12]}...")
print(f"  [{df.index[2]}] {df['load_profile_date'].iloc[2]} | {df['tesisat_no_id'].iloc[2][:12]}...")

df = df.sort_values(['tesisat_no_id', 'load_profile_date']).reset_index(drop=True)

print(f"\nSiralama sonrasi (ilk 3 satirin tarihleri):")
print(f"  [0] {df['load_profile_date'].iloc[0]} | {df['tesisat_no_id'].iloc[0][:12]}...")
print(f"  [1] {df['load_profile_date'].iloc[1]} | {df['tesisat_no_id'].iloc[1][:12]}...")
print(f"  [2] {df['load_profile_date'].iloc[2]} | {df['tesisat_no_id'].iloc[2][:12]}...")

print(f"\nSon 3 satir:")
for i in [-3, -2, -1]:
    print(f"  [{len(df)+i}] {df['load_profile_date'].iloc[i]} | {df['tesisat_no_id'].iloc[i][:12]}...")

# ============================================================
# 4. Abone bazinda zaman atlamasi ve tekrar kontrolu
# ============================================================
print("\n" + "=" * 60)
print("4. ZAMAN ATLAMASI VE TEKRAR KONTROLU")
print("=" * 60)

# 4a: Duplike tarih kontrolu (ayni abone, ayni zaman)
print("--- 4a: Duplike Tarih Kontrolu ---")
duplike = df.duplicated(subset=['tesisat_no_id', 'load_profile_date'], keep=False)
print(f"Ayni abone + ayni tarih-saat tekrari: {duplike.sum():,} satir")

if duplike.sum() > 0:
    dup_df = df[duplike].sort_values(['tesisat_no_id', 'load_profile_date'])
    print(f"Etkilenen abone sayisi: {dup_df['tesisat_no_id'].nunique()}")
    print("Ornekler:")
    for _, row in dup_df.head(6).iterrows():
        print(f"  {row['tesisat_no_id'][:12]}... | {row['load_profile_date']} | t0={row['t0']:.4f}")

# 4b: Zaman farki analizi (abone bazinda)
print("\n--- 4b: Ardisik Kayitlar Arasi Zaman Farki ---")
df['time_diff'] = df.groupby('tesisat_no_id')['load_profile_date'].diff()

# Ilk satirlar NaT olur (her abonenin ilk kaydi)
time_diffs = df['time_diff'].dropna()
print(f"Hesaplanan zaman farki: {len(time_diffs):,} kayit")

# Fark dagilimi
diff_counts = time_diffs.value_counts().head(15)
print(f"\nEn sik zaman farklari:")
for diff, count in diff_counts.items():
    print(f"  {str(diff):>25} -> {count:,} kayit ({count/len(time_diffs)*100:.2f}%)")

# Kategorize et
td_15m = pd.Timedelta(minutes=15)
td_1h = pd.Timedelta(hours=1)

tam_15dk = (time_diffs == td_15m).sum()
tam_1saat = (time_diffs == td_1h).sum()
diger_pozitif = ((time_diffs > pd.Timedelta(0)) & (time_diffs != td_15m) & (time_diffs != td_1h)).sum()
negatif = (time_diffs < pd.Timedelta(0)).sum()
sifir = (time_diffs == pd.Timedelta(0)).sum()

print(f"\nKategori ozeti:")
print(f"  Tam 15 dakika    : {tam_15dk:,} ({tam_15dk/len(time_diffs)*100:.2f}%)")
print(f"  Tam 1 saat       : {tam_1saat:,} ({tam_1saat/len(time_diffs)*100:.2f}%)")
print(f"  Diger (pozitif)  : {diger_pozitif:,} ({diger_pozitif/len(time_diffs)*100:.2f}%)")
print(f"  Sifir (tekrar?)  : {sifir:,}")
print(f"  Negatif (hata?)  : {negatif:,}")

# 4c: 1 saatlik periyotlu aboneleri tespit et
print("\n--- 4c: 1 Saatlik Periyotlu Aboneler ---")
abone_periyot = df.groupby('tesisat_no_id')['time_diff'].agg(
    lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) > 0 and len(x.dropna().mode()) > 0 else pd.NaT
)
saatlik_aboneler = abone_periyot[abone_periyot == td_1h]
on_bes_dk_aboneler = abone_periyot[abone_periyot == td_15m]

print(f"15 dakika periyotlu abone: {len(on_bes_dk_aboneler)}")
print(f"1 saat periyotlu abone   : {len(saatlik_aboneler)}")

if len(saatlik_aboneler) > 0:
    print(f"\n1 saatlik periyotlu aboneler:")
    for tesisat in saatlik_aboneler.index:
        kayit = len(df[df['tesisat_no_id'] == tesisat])
        model = df[df['tesisat_no_id'] == tesisat]['model'].iloc[0]
        gs = df[df['tesisat_no_id'] == tesisat]['gerilim_seviyesi'].iloc[0]
        print(f"  {tesisat[:12]}... | {kayit:,} kayit | {model} | {gs}")

# 4d: Buyuk bosluklar (>1 saat, 15dk aboneler icin / >2 saat, 1h aboneler icin)
print("\n--- 4d: Buyuk Bosluklar ---")
buyuk_bosluk = []
for tesisat in df['tesisat_no_id'].unique():
    sub = df[df['tesisat_no_id'] == tesisat]
    diffs = sub['time_diff'].dropna()
    
    # Bu abonenin beklenen periyodunu belirle
    if tesisat in saatlik_aboneler.index:
        esik = pd.Timedelta(hours=2)
        periyot = "1h"
    else:
        esik = pd.Timedelta(hours=1)
        periyot = "15m"
    
    boslukar = diffs[diffs > esik]
    for gap in boslukar:
        buyuk_bosluk.append({
            'tesisat': tesisat[:12],
            'bosluk': gap,
            'saat': gap.total_seconds() / 3600,
            'periyot': periyot
        })

if len(buyuk_bosluk) > 0:
    bb_df = pd.DataFrame(buyuk_bosluk).sort_values('saat', ascending=False)
    print(f"Toplam buyuk bosluk: {len(bb_df)}")
    print(f"Etkilenen abone   : {bb_df['tesisat'].nunique()}")
    print(f"\nEn buyuk 15 bosluk:")
    for _, row in bb_df.head(15).iterrows():
        print(f"  {row['tesisat']}... | {row['saat']:.1f} saat | periyot: {row['periyot']}")
else:
    print("Buyuk bosluk bulunamadi.")

# ============================================================
# 5. Zaman bazli analiz icin hazirlik degerlendirmesi
# ============================================================
print("\n" + "=" * 60)
print("5. ZAMAN BAZLI ANALIZ HAZIRLIK DEGERLENDIRMESI")
print("=" * 60)

# Abone basina beklenen vs gercek kayit
print("--- Abone Bazinda Veri Tamligi ---")
tarih_min = df['load_profile_date'].min()
tarih_max = df['load_profile_date'].max()

tamlik = []
for tesisat in df['tesisat_no_id'].unique():
    sub = df[df['tesisat_no_id'] == tesisat]
    t_min = sub['load_profile_date'].min()
    t_max = sub['load_profile_date'].max()
    gercek = len(sub)
    
    if tesisat in saatlik_aboneler.index:
        beklenen = int((t_max - t_min).total_seconds() / 3600) + 1
        periyot = "1h"
    else:
        beklenen = int((t_max - t_min).total_seconds() / 900) + 1
        periyot = "15m"
    
    oran = gercek / beklenen * 100 if beklenen > 0 else 0
    tamlik.append({
        'tesisat': tesisat[:12],
        'periyot': periyot,
        'gercek': gercek,
        'beklenen': beklenen,
        'tamlik': oran,
        't_min': t_min,
        't_max': t_max
    })

tamlik_df = pd.DataFrame(tamlik)

print(f"\n  15 dakika periyotlu aboneler ({len(tamlik_df[tamlik_df['periyot']=='15m'])} adet):")
t15 = tamlik_df[tamlik_df['periyot'] == '15m']
print(f"    Tamlik orani: min={t15['tamlik'].min():.1f}% | ort={t15['tamlik'].mean():.1f}% | max={t15['tamlik'].max():.1f}%")
print(f"    <%95 tamlik: {(t15['tamlik'] < 95).sum()} abone")
print(f"    >=%95 tamlik: {(t15['tamlik'] >= 95).sum()} abone")

print(f"\n  1 saat periyotlu aboneler ({len(tamlik_df[tamlik_df['periyot']=='1h'])} adet):")
t1h = tamlik_df[tamlik_df['periyot'] == '1h']
if len(t1h) > 0:
    print(f"    Tamlik orani: min={t1h['tamlik'].min():.1f}% | ort={t1h['tamlik'].mean():.1f}% | max={t1h['tamlik'].max():.1f}%")

# Dusuk tamlikli aboneler
dusuk = tamlik_df[tamlik_df['tamlik'] < 90].sort_values('tamlik')
if len(dusuk) > 0:
    print(f"\n  Dusuk tamlik (<90%) olan aboneler:")
    for _, row in dusuk.iterrows():
        print(f"    {row['tesisat']}... | {row['tamlik']:.1f}% | {row['gercek']}/{row['beklenen']} kayit | "
              f"{row['t_min'].strftime('%Y-%m-%d')} - {row['t_max'].strftime('%Y-%m-%d')} | {row['periyot']}")

print(f"""
SONUC DEGERLENDIRMESI:
============================================================
- Veri tesisat_no_id + load_profile_date bazinda siralanmistir.
- Duplike kayit durumu yukarida raporlanmistir.
- 2 farkli periyot tespit edilmistir: 15 dakika ve 1 saat.
- 15dk abonelerin buyuk cogunlugu yuksek tamlik oranina sahiptir.
- 1 saatlik aboneler ve dusuk tamlikli aboneler vardir.
- Buyuk bosluklar sinirli sayida abonede gorulmektedir.
- Veri zaman bazli analiz icin kullanimda bu farkliliklar
  goz onunde bulundurulmalidir.
""")

# time_diff kolonunu dusur ve kaydet
df = df.drop(columns=['time_diff'])
df.to_csv("dataset_clean.csv")
print("Siralanmis veri kaydedildi: dataset_clean.csv")
