# LSTM Forex Signal Classifier – EUR/USD

**Skripsi:** Implementasi Model Long Short-Term Memory (LSTM) untuk Klasifikasi Sinyal Trading Forex Berbasis Indikator Teknikal Multi-Timeframe pada Pasangan Mata Uang EUR/USD

**Penulis:** Ahmad Fauzan Abdurrohman (22110007)  
**Program Studi:** Teknik Informatika – STMIK Mardira Indonesia  
**Tahun:** 2026

---

## Struktur Proyek

```
lstm_forex/
├── data_pipeline.py     # Download, preprocessing, labeling, feature engineering
├── lstm_model.py        # Arsitektur LSTM, training, evaluasi
├── app.py               # Web dashboard Flask
├── run_all.py           # Runner pipeline lengkap (1 klik)
├── requirements.txt     # Dependensi Python
├── templates/
│   └── index.html       # Dashboard UI
├── data/                # Dataset (otomatis dibuat)
├── models/              # Model tersimpan (otomatis dibuat)
└── outputs/             # Grafik & metrik (otomatis dibuat)
```

---

## Cara Menjalankan

### 1. Install Dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan Pipeline Lengkap
```bash
python run_all.py
```
Pipeline ini akan:
- Download data EUR/USD dari yfinance (Daily + H4)
- Hitung MA 200 dan fitur teknikal
- Generate label sinyal (BUY / SELL / NO-ENTRY)
- Merge multi-timeframe (D + H4)
- Buat sequences untuk LSTM (sliding window 30 candle)
- Split data 70/15/15 (train/val/test)
- Train 3 varian LSTM: Vanilla, Stacked, Bi-LSTM
- Evaluasi dengan accuracy, precision, recall, F1-score
- Simpan grafik training history & confusion matrix

### 3. Buka Dashboard
```bash
python -m streamlit run app.py
```
Buka browser: **http://localhost:5000**

---

## Spesifikasi Teknis

| Parameter       | Nilai                    |
|-----------------|--------------------------|
| Pasangan Mata Uang | EUR/USD               |
| Timeframe       | Daily (D) + 4 Jam (H4)  |
| Indikator       | Moving Average 200 (MA 200) |
| Periode Data    | Jan 2024 – Mei 2026      |
| Sequence Length | 30 candle (lookback)     |
| Split Ratio     | 70% / 15% / 15%          |
| Kelas Output    | BUY (2) / NO-ENTRY (1) / SELL (0) |

## Arsitektur Model (Stacked LSTM)

```
Input  (30, 16 fitur)
  ↓
LSTM (128 units, return_sequences=True)
  ↓
BatchNormalization + Dropout(0.3)
  ↓
LSTM (64 units)
  ↓
BatchNormalization + Dropout(0.3)
  ↓
Dense (64, relu)
  ↓
Dropout(0.2)
  ↓
Dense (32, relu)
  ↓
Dense (3, softmax) → BUY / SELL / NO-ENTRY
```

## Fitur Input (16 Fitur)

**H4 Features:**
- OHLC (open, high, low, close)
- MA 200
- price_vs_ma200 (% jarak harga dari MA200)
- ma200_slope (momentum MA200)
- body, upper_shadow, lower_shadow
- hl_range (volatilitas)
- return_1, return_5 (momentum return)

**Daily Context Features:**
- d_ma200
- d_price_vs_ma200
- d_ma200_slope

---

## Metode Pengembangan

Menggunakan **CRISP-DM** (Cross-Industry Standard Process for Data Mining):
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment (Web Dashboard)

---

## Output

- `outputs/training_*.png` – Grafik training history
- `outputs/confusion_*.png` – Confusion matrix
- `outputs/comparison.png` – Perbandingan varian LSTM
- `outputs/metrics_*.json` – Metrik evaluasi per varian
- `models/lstm_*.keras` – Model tersimpan