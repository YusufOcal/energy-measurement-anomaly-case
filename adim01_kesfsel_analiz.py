"""
ADIM 1: Veri Setini Yukle ve Kesifsel Analiz
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. CSV dosyasini pandas ile yukle
# ============================================================
df = pd.read_csv("dataset.csv", index_col=0)

# ============================================================
# 2. Satir ve kolon sayisini dogrula
# ============================================================
print("=" * 60)
print("2. SATIR VE KOLON SAYISI")
print("=" * 60)
print(f"Satir sayisi : {df.shape[0]:,}")
print(f"Kolon sayisi : {df.shape[1]}")
# README'de belirtilen: 353,949 satir, 18 kolon
print(f"\nREADME dogrulamasi:")
print(f"  Satir -> README: 353,949 | Gercek: {df.shape[0]:,} | {'UYUMLU' if df.shape[0] == 353949 else 'UYUMSUZ'}")
print(f"  Kolon -> README: 18       | Gercek: {df.shape[1]}    | {'UYUMLU' if df.shape[1] == 18 else 'UYUMSUZ'}")

# ============================================================
# 3. Kolon isimleri ve veri tipleri
# ============================================================
print("\n" + "=" * 60)
print("3. KOLON ISIMLERI VE VERI TIPLERI")
print("=" * 60)
print(f"{'#':<4} {'Kolon Adi':<25} {'Veri Tipi':<15}")
print("-" * 44)
for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
    print(f"{i:<4} {col:<25} {str(dtype):<15}")

# ============================================================
# 4. Ilk 5 ve son 5 satir
# ============================================================
print("\n" + "=" * 60)
print("4a. ILK 5 SATIR")
print("=" * 60)
print(df.head().to_string())

print("\n" + "=" * 60)
print("4b. SON 5 SATIR")
print("=" * 60)
print(df.tail().to_string())

# ============================================================
# 5. Eksik deger oranlari (yuzde)
# ============================================================
print("\n" + "=" * 60)
print("5. EKSIK DEGER ORANLARI")
print("=" * 60)
print(f"{'Kolon':<25} {'Eksik Sayi':>12} {'Eksik %':>10} {'Dolu Sayi':>12}")
print("-" * 59)
for col in df.columns:
    eksik = df[col].isnull().sum()
    oran = eksik / len(df) * 100
    dolu = len(df) - eksik
    print(f"{col:<25} {eksik:>12,} {oran:>9.2f}% {dolu:>12,}")

print(f"\n{'TOPLAM':<25} {df.isnull().sum().sum():>12,}")

# ============================================================
# 6. Tarih alaninin mevcut formatini analiz et
# ============================================================
print("\n" + "=" * 60)
print("6. TARIH ALANI FORMAT ANALIZI (load_profile_date)")
print("=" * 60)
print(f"Mevcut veri tipi  : {df['load_profile_date'].dtype}")
print(f"Ornek degerler    :")
for i in [0, 1, 2, -2, -1]:
    val = df['load_profile_date'].iloc[i]
    print(f"  [{i:>6}] -> '{val}'")

print(f"\nTekil deger sayisi: {df['load_profile_date'].nunique():,}")
print(f"Null deger sayisi : {df['load_profile_date'].isnull().sum()}")

# Format tespiti
sample = str(df['load_profile_date'].iloc[0])
print(f"\nFormat tespiti:")
print(f"  Ornek         : '{sample}'")
print(f"  Uzunluk       : {len(sample)} karakter")
print(f"  Gozlenen format: YYYY-MM-DD HH:MM:SS.mmm")
print(f"  Milisaniye var : {'Evet (.000)' if '.' in sample else 'Hayir'}")
print(f"  Henuz parse edilmedi (dtype = object)")
