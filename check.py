import pandas as pd

# Baca semua file
daily = pd.read_csv('data/eurusd_daily.csv')
h4 = pd.read_csv('data/eurusd_h4.csv')
merged = pd.read_csv('data/dataset_merged.csv')

print("=" * 60)
print("CEK SUMBER OHLC DI MERGED.CSV")
print("=" * 60)

# Cari tanggal yang sama di merged dan h4
merged_date = '2021-10-14 00:00:00'
h4_date = '2021-01-01 00:00:00'

print(f"\nMerged CSV pada {merged_date}:")
merged_row = merged[merged['datetime'] == merged_date]
if not merged_row.empty:
    print(f"  Open:  {merged_row['open'].values[0]}")
    print(f"  High:  {merged_row['high'].values[0]}")
    print(f"  Low:   {merged_row['low'].values[0]}")
    print(f"  Close: {merged_row['close'].values[0]}")

print(f"\nH4 CSV pada {h4_date}:")
h4_row = h4[h4['datetime'] == h4_date]
if not h4_row.empty:
    print(f"  Open:  {h4_row['open'].values[0]}")
    print(f"  High:  {h4_row['high'].values[0]}")
    print(f"  Low:   {h4_row['low'].values[0]}")
    print(f"  Close: {h4_row['close'].values[0]}")

# Cek apakah ada data H4 yang sama dengan merged
print("\n" + "=" * 60)
print("MENCARI DATA YANG SAMA DI H4.CSV")
print("=" * 60)

for idx, row in h4.iterrows():
    if abs(row['open'] - merged_row['open'].values[0]) < 0.0001:
        print(f"DITEMUKAN! di H4 index {idx}:")
        print(f"  datetime: {row['datetime']}")
        print(f"  open: {row['open']}")
        break
else:
    print("TIDAK DITEMUKAN data yang sama di H4.csv")
    print("Kesimpulan: OHLC di merged.csv BUKAN dari H4.csv!")

# Cek apakah dari Daily
print("\n" + "=" * 60)
print("MENCARI DI DAILY.CSV")
print("=" * 60)

for idx, row in daily.iterrows():
    if abs(row['open'] - merged_row['open'].values[0]) < 0.01:
        print(f"MENDekati! Daily index {idx}:")
        print(f"  Date: {row['Date']}")
        print(f"  open: {row['open']}")
        break