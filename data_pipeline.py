"""
data_pipeline.py
Pengumpulan & Preprocessing Data EUR/USD Multi-Timeframe
Skripsi: Implementasi LSTM untuk Klasifikasi Sinyal Trading Forex
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle, os, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

PAIR         = "EURUSD=X"
START_DATE   = "2021-01-01"
END_DATE     = datetime.now().strftime("%Y-%m-%d")
MA_PERIOD    = 200
SEQUENCE_LEN = 30
DATA_DIR     = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DOWNLOAD / GENERATE DATA
# ─────────────────────────────────────────────
def download_data():
    print("=" * 55)
    print(f"  DATA EUR/USD ({START_DATE} – {END_DATE})")
    print("=" * 55)

    # Coba yfinance dulu
    try:
        import yfinance as yf
        print("\n[1/2] Mengunduh timeframe Daily (D)...")
        df_d = yf.download(PAIR, start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, progress=False)
        df_d = df_d[["Open","High","Low","Close"]].copy()
        df_d.columns = ["open","high","low","close"]
        df_d.dropna(inplace=True)
        if len(df_d) < 50:
            raise ValueError("Data tidak cukup")
        print(f"      → {len(df_d)} candle daily diperoleh.")
    except Exception as e:
        print(f"      ⚠  yfinance gagal ({e.__class__.__name__}), menggunakan data simulasi realistis...")
        df_d = _generate_realistic_eurusd_daily()
        print(f"      → {len(df_d)} candle daily (simulasi) dibuat.")

    print("[2/2] Membuat timeframe H4 dari Daily...")
    df_h4 = _synthetic_h4_from_daily(df_d)
    print(f"      → {len(df_h4)} candle H4 dibuat.")

    df_d.to_csv(f"{DATA_DIR}/eurusd_daily.csv")
    df_h4.to_csv(f"{DATA_DIR}/eurusd_h4.csv")
    print(f"\n✓ Data disimpan ke {DATA_DIR}/")
    return df_d, df_h4


def _generate_realistic_eurusd_daily():
    """Simulasi data EUR/USD realistis dengan geometric Brownian motion + trend."""
    np.random.seed(42)
    dates = pd.bdate_range(start=START_DATE, end=END_DATE, freq='B')
    n = len(dates)

    # GBM parameters (EUR/USD realistis)
    S0    = 1.1850    # harga awal Jan 2021
    mu    = -0.00005    # drift harian sedikit negatif (periode 2021-2022 bearish)
    sigma = 0.0065    # volatilitas harian EUR/USD ~0.65%

    returns = np.random.normal(mu, sigma, n)
    # Tren makro untuk periode 2021-2026 (5.5 tahun)
    # 2021: naik, 2022-2023: turun karena perang, 2024-2025: naik, 2026: konsolidasi
    n1 = n // 5   # 2021
    n2 = n1 * 2   # 2022-2023  
    n3 = n1 * 2   # 2024-2025
    n4 = n - (n1 + n2 + n3)  # 2026

    trend = np.concatenate([
        np.linspace(0, 0.06, n1),           # 2021 naik 6%
        np.linspace(0.06, -0.08, n2),       # 2022-2023 turun 14% dari puncak
        np.linspace(-0.08, 0.05, n3),       # 2024-2025 naik 13% dari bottom
        np.linspace(0.05, -0.02, n4)        # 2026 konsolidasi turun 7%
    ])
    close = S0 * np.exp(np.cumsum(returns)) + trend

    # OHLC dari close
    noise = np.abs(np.random.normal(0, 0.0030, n))
    high  = close + noise * np.random.uniform(0.4, 1.0, n)
    low   = close - noise * np.random.uniform(0.4, 1.0, n)
    open_ = close - np.random.normal(0, 0.0020, n)
    open_ = np.clip(open_, low, high)

    df = pd.DataFrame({
        "open": np.round(open_, 5),
        "high": np.round(high, 5),
        "low":  np.round(low, 5),
        "close": np.round(close, 5)
    }, index=dates)
    df.index.name = "datetime"
    return df


def _synthetic_h4_from_daily(df_d):
    """Buat H4 sintetis: tiap hari Daily → 6 candle H4."""
    np.random.seed(99)
    rows = []
    for ts, row in df_d.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        prices = np.linspace(o, c, 7)
        for i in range(6):
            noise  = np.random.uniform(-0.0008, 0.0008)
            h4_o   = float(prices[i])
            h4_c   = float(prices[i+1]) + noise
            h4_h   = min(max(h4_o, h4_c) + abs(noise)*1.5, float(h))
            h4_l   = max(min(h4_o, h4_c) - abs(noise)*1.5, float(l))
            rows.append({
                "open":  round(h4_o, 5), "high": round(h4_h, 5),
                "low":   round(h4_l, 5), "close": round(h4_c, 5)
            })
    ts_index = []
    for ts in df_d.index:
        for i in range(6):
            ts_index.append(pd.Timestamp(ts) + pd.Timedelta(hours=i*4))
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(ts_index))
    df.index.name = "datetime"
    return df


# ─────────────────────────────────────────────
# 2. KALKULASI MA 200
# ─────────────────────────────────────────────
def compute_ma200(df):
    df = df.copy()
    df["ma200"]        = df["close"].rolling(MA_PERIOD).mean()
    df["price_vs_ma200"] = (df["close"] - df["ma200"]) / df["ma200"] * 100
    df["ma200_slope"]  = df["ma200"].diff(5) / df["ma200"].shift(5) * 100
    df["body"]         = df["close"] - df["open"]
    df["upper_shadow"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_shadow"] = df[["open","close"]].min(axis=1) - df["low"]
    df["hl_range"]     = df["high"] - df["low"]
    df["return_1"]     = df["close"].pct_change(1)
    df["return_5"]     = df["close"].pct_change(5)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3. LABELING
# ─────────────────────────────────────────────
def generate_labels(df, forward_period=5, threshold=0.0015):
    df = df.copy()
    future_return = df["close"].shift(-forward_period) / df["close"] - 1
    conditions = [
        (future_return >  threshold) & (df["close"] > df["ma200"]),
        (future_return < -threshold) & (df["close"] < df["ma200"]),
    ]
    df["signal"] = np.select(conditions, [2, 0], default=1)
    df.dropna(inplace=True)
    df = df.iloc[:-forward_period]
    return df


# ─────────────────────────────────────────────
# 4. MERGE TIMEFRAMES
# ─────────────────────────────────────────────
def merge_timeframes(df_d, df_h4):
    df_d_feat  = compute_ma200(df_d)
    df_h4_feat = compute_ma200(df_h4)
    df_h4_feat = generate_labels(df_h4_feat)

    daily_ctx = df_d_feat[["ma200","price_vs_ma200","ma200_slope"]].copy()
    daily_ctx.index = pd.to_datetime(daily_ctx.index).normalize()
    daily_ctx.columns = ["d_ma200","d_price_vs_ma200","d_ma200_slope"]

    h4_idx = pd.to_datetime(df_h4_feat.index)
    df_h4_feat.index = h4_idx
    df_h4_feat["date_key"] = h4_idx.normalize()

    merged = df_h4_feat.merge(daily_ctx, left_on="date_key", right_index=True, how="left")
    merged.drop(columns=["date_key"], inplace=True)
    merged.ffill(inplace=True)
    merged.dropna(inplace=True)

    print(f"\n✓ Dataset gabungan: {len(merged)} sampel")
    dist = merged["signal"].value_counts().sort_index()
    labels = {0:"SELL", 1:"NO-ENTRY", 2:"BUY"}
    for k, v in dist.items():
        print(f"   {labels[k]:10s}: {v:5d} ({v/len(merged)*100:.1f}%)")

    merged.to_csv(f"{DATA_DIR}/dataset_merged.csv")
    return merged


# ─────────────────────────────────────────────
# 5. NORMALISASI & SEQUENCES
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "open","high","low","close",
    "ma200","price_vs_ma200","ma200_slope",
    "body","upper_shadow","lower_shadow",
    "hl_range","return_1","return_5",
    "d_ma200","d_price_vs_ma200","d_ma200_slope"
]

def build_sequences(df, seq_len=SEQUENCE_LEN):
    scaler = MinMaxScaler()
    features = df[FEATURE_COLS].values
    features_scaled = scaler.fit_transform(features)
    labels = df["signal"].values

    X, y = [], []
    for i in range(seq_len, len(features_scaled)):
        X.append(features_scaled[i-seq_len:i])
        y.append(labels[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    with open(f"{DATA_DIR}/scaler.pkl","wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✓ Sequences: X={X.shape}, y={y.shape}")
    return X, y, scaler


# ─────────────────────────────────────────────
# 6. SPLIT DATA
# ─────────────────────────────────────────────
def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n  = len(X)
    i1 = int(n*train_ratio)
    i2 = int(n*(train_ratio+val_ratio))
    X_train,y_train = X[:i1],  y[:i1]
    X_val,  y_val   = X[i1:i2],y[i1:i2]
    X_test, y_test  = X[i2:],  y[i2:]
    print(f"\n✓ Split 70/15/15:")
    print(f"   Train : {len(X_train):5d}")
    print(f"   Val   : {len(X_val):5d}")
    print(f"   Test  : {len(X_test):5d}")
    return (X_train,y_train),(X_val,y_val),(X_test,y_test)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_pipeline():
    df_d, df_h4 = download_data()
    merged = merge_timeframes(df_d, df_h4)
    X, y, scaler = build_sequences(merged)
    splits = split_data(X, y)
    np.save(f"{DATA_DIR}/X_train.npy", splits[0][0])
    np.save(f"{DATA_DIR}/y_train.npy", splits[0][1])
    np.save(f"{DATA_DIR}/X_val.npy",   splits[1][0])
    np.save(f"{DATA_DIR}/y_val.npy",   splits[1][1])
    np.save(f"{DATA_DIR}/X_test.npy",  splits[2][0])
    np.save(f"{DATA_DIR}/y_test.npy",  splits[2][1])
    print(f"\n✓ Semua data disimpan di {DATA_DIR}/")
    return splits, scaler, merged

if __name__ == "__main__":
    run_pipeline()