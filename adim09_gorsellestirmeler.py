"""
ADIM 9: Yonetici Odakli Gorsellestirmeler
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# Font ve stil ayarlari
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
})

df = pd.read_csv("dataset_clean.csv", index_col=0)
df['load_profile_date'] = pd.to_datetime(df['load_profile_date'])

anomali_kolonlari = [
    'A1_Akim_Var_Tuketim_Yok', 'A2_Tuketim_Var_Akim_Dusuk',
    'A3_Faz_Dengesizligi_Yuksek', 'A4_Gerilim_Eksik_Tuketim_Var',
    'A5_Sifir_Negatif_Tuketim', 'A6_Sabit_Tuketim',
    'A7_Gece_Olagandisi', 'A8_Yuksek_Reaktif'
]

anomali_kisa = [
    'A1\nAkim Var\nTuketim Yok',
    'A2\nTuketim Var\nAkim Dusuk',
    'A3\nFaz\nDengesizligi',
    'A4\nGerilim\nEksik',
    'A5\nSifir\nTuketim',
    'A6\nSabit\nTuketim',
    'A7\nGece\nAnomali',
    'A8\nYuksek\nReaktif'
]

# Renk paleti
renkler = ['#E74C3C', '#E67E22', '#F1C40F', '#3498DB',
           '#9B59B6', '#1ABC9C', '#2C3E50', '#E84393']

# ============================================================
# GRAFIK 1: Anomali Turu Dagilimi (Yatay Bar)
# ============================================================
print("Grafik 1: Anomali turu dagilimi olusturuluyor...")

fig, ax = plt.subplots(figsize=(12, 6))

counts = [df[col].sum() for col in anomali_kolonlari]
labels = [
    'A1 - Akim Var / Tuketim Yok',
    'A2 - Tuketim Var / Akim Dusuk',
    'A3 - Faz Dengesizligi + Yuksek Akim',
    'A4 - Gerilim Eksik / Tuketim Var',
    'A5 - Sifir veya Negatif Tuketim',
    'A6 - Uzun Sure Sabit Tuketim',
    'A7 - Gece Saati Olagandisi Tuketim',
    'A8 - Yuksek Reaktif / Dusuk Aktif'
]

# Sirala (buyukten kucuge)
sorted_pairs = sorted(zip(counts, labels, renkler), key=lambda x: x[0])
s_counts, s_labels, s_colors = zip(*sorted_pairs)

bars = ax.barh(range(len(s_counts)), s_counts, color=s_colors, edgecolor='white', height=0.65)

# Deger etiketleri
for i, (bar, count) in enumerate(zip(bars, s_counts)):
    if count > 0:
        pct = count / len(df) * 100
        ax.text(bar.get_width() + 800, bar.get_y() + bar.get_height()/2,
                f'{count:,}  (%{pct:.1f})', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(s_labels)))
ax.set_yticklabels(s_labels, fontsize=9)
ax.set_xlabel('Kayit Sayisi')
ax.set_title('Anomali Turlerine Gore Dagilim')
ax.set_xlim(0, max(s_counts) * 1.25)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Toplam bilgi kutusu
toplam_anomalili = (df['Toplam_Anomali_Sayisi'] > 0).sum()
ax.text(0.98, 0.05,
        f'Toplam Kayit: {len(df):,}\nAnomalili: {toplam_anomalili:,} (%{toplam_anomalili/len(df)*100:.1f})\nTemiz: {len(df)-toplam_anomalili:,} (%{(len(df)-toplam_anomalili)/len(df)*100:.1f})',
        transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

plt.tight_layout()
plt.savefig('grafik1_anomali_dagilimi.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Kaydedildi: grafik1_anomali_dagilimi.png")


# ============================================================
# GRAFIK 2: Gunluk Anomali Trendi (Time Series)
# ============================================================
print("Grafik 2: Gunluk anomali trendi olusturuluyor...")

df['tarih'] = df['load_profile_date'].dt.date
daily = df.groupby('tarih').agg(
    toplam=('Toplam_Anomali_Sayisi', 'size'),
    anomalili=('Toplam_Anomali_Sayisi', lambda x: (x > 0).sum()),
)
daily['oran'] = daily['anomalili'] / daily['toplam'] * 100
daily.index = pd.to_datetime(daily.index)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

# Ust panel: Anomalili kayit sayisi
ax1.fill_between(daily.index, daily['anomalili'], alpha=0.3, color='#E74C3C')
ax1.plot(daily.index, daily['anomalili'], color='#E74C3C', linewidth=1.5, label='Anomalili Kayit')
ax1.plot(daily.index, daily['toplam'], color='#2C3E50', linewidth=1, alpha=0.5, linestyle='--', label='Toplam Kayit')

# Hareketli ortalama
if len(daily) >= 7:
    ma7 = daily['anomalili'].rolling(7, center=True).mean()
    ax1.plot(daily.index, ma7, color='#E74C3C', linewidth=2.5, label='7 Gunluk Ort.', zorder=5)

ax1.set_ylabel('Kayit Sayisi')
ax1.set_title('Gunluk Anomali Trendi (Agustos - Ekim 2025)')
ax1.legend(loc='upper right', fontsize=8)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Alt panel: Anomali orani
ax2.bar(daily.index, daily['oran'], color='#3498DB', alpha=0.7, width=0.8)
ax2.axhline(y=daily['oran'].mean(), color='#E74C3C', linestyle='--', linewidth=1.5,
            label=f'Ortalama: %{daily["oran"].mean():.1f}')
ax2.set_ylabel('Anomali Orani (%)')
ax2.set_xlabel('Tarih')
ax2.legend(loc='upper right', fontsize=8)
ax2.set_ylim(0, 50)

# Ay sinirlari
for ay_baslangic in ['2025-09-01', '2025-10-01']:
    for ax in [ax1, ax2]:
        ax.axvline(x=pd.Timestamp(ay_baslangic), color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('grafik2_gunluk_anomali_trendi.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Kaydedildi: grafik2_gunluk_anomali_trendi.png")


# ============================================================
# GRAFIK 3: Il Bazinda Anomali Yogunlugu
# ============================================================
print("Grafik 3: Il bazinda anomali yogunlugu olusturuluyor...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Sol: Il bazinda anomali oran
il_data = df.groupby('il').agg(
    toplam=('Toplam_Anomali_Sayisi', 'size'),
    anomalili=('Toplam_Anomali_Sayisi', lambda x: (x > 0).sum()),
    abone=('tesisat_no_id', 'nunique'),
)
il_data['oran'] = il_data['anomalili'] / il_data['toplam'] * 100
il_data = il_data.sort_values('oran', ascending=True)

il_renk = ['#27AE60' if o < 30 else '#F39C12' if o < 50 else '#E74C3C' for o in il_data['oran']]

bars = ax1.barh(il_data.index.astype(str), il_data['oran'], color=il_renk, edgecolor='white', height=0.55)

for i, (bar, row) in enumerate(zip(bars, il_data.itertuples())):
    ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'%{row.oran:.1f}  ({row.abone} abone)', va='center', fontsize=9)

ax1.set_xlabel('Anomali Orani (%)')
ax1.set_title('Il Bazinda Anomali Orani')
ax1.set_xlim(0, 85)
ax1.axvline(x=50, color='red', linestyle='--', alpha=0.3, label='%50 esik')

# Sag: Il bazinda anomali tipi yigilmali bar
il_tip = df.groupby('il')[anomali_kolonlari].sum().astype(int)
il_tip = il_tip.loc[il_data.index]

bottom = np.zeros(len(il_tip))
anomali_kisa_bar = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']

for i, (col, kisa, renk) in enumerate(zip(anomali_kolonlari, anomali_kisa_bar, renkler)):
    vals = il_tip[col].values
    ax2.barh(il_tip.index.astype(str), vals, left=bottom, color=renk, label=kisa, 
             edgecolor='white', linewidth=0.3, height=0.55)
    bottom += vals

ax2.set_xlabel('Anomali Kayit Sayisi')
ax2.set_title('Il Bazinda Anomali Tipi Dagilimi')
ax2.legend(loc='lower right', fontsize=7, ncol=4)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout()
plt.savefig('grafik3_il_bazinda_anomali.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Kaydedildi: grafik3_il_bazinda_anomali.png")


# ============================================================
# GRAFIK 4: Yuksek Riskli Tesisatlarin Gorsel Ozeti
# ============================================================
print("Grafik 4: Yuksek riskli tesisatlar olusturuluyor...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Sol: Ilk 10 tesisat anomali puani ---
tesisat_data = df.groupby('tesisat_no_id').agg(
    toplam_anomali=('Toplam_Anomali_Sayisi', 'sum'),
    kayit=('Toplam_Anomali_Sayisi', 'size'),
    anomalili=('Toplam_Anomali_Sayisi', lambda x: (x > 0).sum()),
    il=('il', 'first'),
    model=('model', 'first'),
    gerilim=('gerilim_seviyesi', 'first'),
)
tesisat_data['oran'] = (tesisat_data['anomalili'] / tesisat_data['kayit'] * 100).round(1)
top10 = tesisat_data.nlargest(10, 'toplam_anomali')

# Kisa etiketler
y_labels = []
for idx, row in top10.iterrows():
    y_labels.append(f"{idx[:8]}.. | {row['il']}")

# Anomali tipi kirilimli yigilmali bar
bottom = np.zeros(10)
for i, (col, kisa, renk) in enumerate(zip(anomali_kolonlari, anomali_kisa_bar, renkler)):
    vals = []
    for tesisat in top10.index:
        vals.append(df[df['tesisat_no_id'] == tesisat][col].sum())
    vals = np.array(vals)
    ax1.barh(range(10), vals, left=bottom, color=renk, label=kisa, 
             edgecolor='white', linewidth=0.3, height=0.6)
    bottom += vals

ax1.set_yticks(range(10))
ax1.set_yticklabels(y_labels, fontsize=8)
ax1.set_xlabel('Toplam Anomali Puani')
ax1.set_title('En Riskli 10 Tesisat - Anomali Kirilimi')
ax1.legend(loc='lower right', fontsize=7, ncol=4)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax1.invert_yaxis()

# --- Sag: Ilk 10 tesisat - anomali heatmap ---
# Her tesisat icin anomali orani (%)
heat_data = []
for tesisat in top10.index:
    sub = df[df['tesisat_no_id'] == tesisat]
    row_data = []
    for col in anomali_kolonlari:
        row_data.append(sub[col].mean() * 100)
    heat_data.append(row_data)

heat_df = pd.DataFrame(heat_data, index=[f"{t[:8]}.." for t in top10.index], 
                        columns=anomali_kisa_bar)

im = ax2.imshow(heat_df.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

# Etiketler
ax2.set_xticks(range(len(anomali_kisa_bar)))
ax2.set_xticklabels(anomali_kisa_bar, fontsize=9)
ax2.set_yticks(range(len(heat_df)))
ax2.set_yticklabels(heat_df.index, fontsize=8)
ax2.set_title('En Riskli 10 Tesisat - Anomali Yogunluk Haritasi (%)')

# Hucre degerleri
for i in range(len(heat_df)):
    for j in range(len(anomali_kisa_bar)):
        val = heat_df.values[i, j]
        if val > 0.5:
            color = 'white' if val > 50 else 'black'
            ax2.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, 
                    fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
cbar.set_label('Anomali Orani (%)', fontsize=9)

plt.tight_layout()
plt.savefig('grafik4_yuksek_riskli_tesisatlar.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Kaydedildi: grafik4_yuksek_riskli_tesisatlar.png")


# ============================================================
# OZET
# ============================================================
print(f"\n{'='*60}")
print("OLUSTURULAN GORSELLER")
print(f"{'='*60}")
print(f"  1. grafik1_anomali_dagilimi.png        - Anomali turu dagilimi (yatay bar)")
print(f"  2. grafik2_gunluk_anomali_trendi.png   - Gunluk anomali trendi (cift panel)")
print(f"  3. grafik3_il_bazinda_anomali.png      - Il bazinda oran + tip kirilimi")
print(f"  4. grafik4_yuksek_riskli_tesisatlar.png - Ilk 10 tesisat kirilim + heatmap")
