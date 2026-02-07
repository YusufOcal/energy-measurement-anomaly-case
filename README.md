# ⚡ Elektrik Ölçüm Verileri (OG Hatları)

Bu veri seti, YEDAŞ elektrik dağıtım hizmeti kapsamındaki abonelere ait 15 dakikalık periyotlarla kaydedilmiş elektrik ölçüm verilerini içermektedir.
Her satır bir zaman aralığına ait faz, voltaj ve enerji sayaç bilgilerini temsil eder.


## 📈 Veri Özeti

- Toplam Satır Sayısı: 353,949
- Kolon Sayısı: 18


## 🧱 Kolon Açıklamaları

| Kolon Adı | Veri Tipi | Eksik (%) | Açıklama |
|------------|------------|------------|--------------|
| tesisat_no_id | object | %0.0 | Abonenin tesisat bilgileri |
| il | object | %0.0 | Abonenin bukunduğu il. |
| ilce | object | %0.0 | Abonenin bulunduğu ilçe. |
| gerilim_seviyesi | object | %0.0 | Abonenin bağlı olduğu elektrik gerilim seviyesi |
| marka | object | %0.0 | Abonenin elektrik sayacı markası. |
| model | object | %0.0 | Abonenin elektrik sayacı modeli. |
| abone_grubu | object | %0.0 | Abonenin grubu. |
| son_carpan_degeri | int64 | %0.0 | Sayacın son çarpan değeri (ölçümlerin gerçek enerjiye çevrilmesinde kullanılan katsayı). |
| l1 | float64 | %0.0 | L1 fazındaki akım değeri (A). |
| l2 | float64 | %0.0 | L2 fazındaki akım değeri (A). |
| l3 | float64 | %0.0 | L3 fazındaki akım değeri (A). |
| v1 | float64 | %21.3 | L1 fazındaki gerilim değeri (V) |
| v2 | float64 | %21.3 | L2 fazındaki gerilim değeri (V) |
| v3 | float64 | %21.3 | L3 fazındaki gerilim değeri (V)  |
| t0 | float64 | %0.0 | Toplam enerji tüketimi (kWh) veya ana sayaç değeri. |
| ri | float64 | %2.9 | Reaktif enerji değeri (indüktif) veya hatalı güç göstergesi. |
| rc | float64 | %2.9 | Reaktif enerji değeri (kapasitif) veya hatalı güç göstergesi. |
| load_profile_date | object | %0.0 | Ölçümün yapıldığı tarih (YYYY-MM-DD formatında). |

## 📊 Örnek Satırlar

| tesisat_no_id                    | il   | ilce   | gerilim_seviyesi   | marka   | model         | abone_grubu                |   son_carpan_degeri |   l1 |    l2 |   l3 |     v1 |     v2 |     v3 |     t0 |      ri |      rc | load_profile_date       |
|:---------------------------------|:-----|:-------|:-------------------|:--------|:--------------|:---------------------------|--------------------:|-----:|------:|-----:|-------:|-------:|-------:|-------:|--------:|--------:|:------------------------|
| 208e38ba076e428c9e5c9d9c0bd53f3f | ORDU | Gölköy | OG                 | MAKEL   | C500.KMY.2251 | Tek Terimli Ticarethane OG |                   1 |    0 | 64.67 | 0.01 | 233.75 | 222.62 | 232.08 | 415006 | 1019.78 | 2030    | 2025-08-01 00:00:00.000 |
| 208e38ba076e428c9e5c9d9c0bd53f3f | ORDU | Gölköy | OG                 | MAKEL   | C500.KMY.2251 | Tek Terimli Ticarethane OG |                   1 |    0 | 60.41 | 0.01 | 233.68 | 222.3  | 230.85 | 415009 | 1019.78 | 2030.01 | 2025-08-01 00:15:00.000 |
| 208e38ba076e428c9e5c9d9c0bd53f3f | ORDU | Gölköy | OG                 | MAKEL   | C500.KMY.2251 | Tek Terimli Ticarethane OG |                   1 |    0 | 63.23 | 0.01 | 233.48 | 222.11 | 230.98 | 415013 | 1019.78 | 2030.02 | 2025-08-01 00:30:00.000 |
| 208e38ba076e428c9e5c9d9c0bd53f3f | ORDU | Gölköy | OG                 | MAKEL   | C500.KMY.2251 | Tek Terimli Ticarethane OG |                   1 |    0 | 65    | 0.01 | 235.23 | 222.52 | 232    | 415016 | 1019.78 | 2030.03 | 2025-08-01 00:45:00.000 |
| 208e38ba076e428c9e5c9d9c0bd53f3f | ORDU | Gölköy | OG                 | MAKEL   | C500.KMY.2251 | Tek Terimli Ticarethane OG |                   1 |    0 | 62.46 | 0.01 | 234.7  | 222.52 | 231.78 | 415019 | 1019.78 | 2030.04 | 2025-08-01 01:00:00.000 |

## 💡 Olası Kullanım Alanları
- Elektrik tüketim analizleri
- Enerji talep tahmini (time series)
- Anomali tespiti ve yük profili çıkarımı
- Sayaç veri doğrulama (validation)

## ⚙️ Veri Kaynağı ve Lisans
- Veri kaynağı: Anonimleştirilmiş OSOS ölçüm verileri
- Tarih: 2025-10-24

