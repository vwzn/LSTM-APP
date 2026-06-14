"""
run_all.py
Pipeline Lengkap: Download → Preprocess → Train → Evaluate → Compare
Jalankan: python run_all.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("""
╔══════════════════════════════════════════════════════╗
║  LSTM FOREX SIGNAL CLASSIFIER — EUR/USD              ║
║  Skripsi: Ahmad Fauzan Abdurrohman (22110007)        ║
║  STMIK Mardira Indonesia · 2026                      ║
╚══════════════════════════════════════════════════════╝
""")

# ────────────────────────────────────────────
# STEP 1: DATA PIPELINE
# ────────────────────────────────────────────
print("\n[STEP 1/3] DATA PIPELINE")
print("─" * 40)
from data_pipeline import run_pipeline
splits, scaler, merged_df = run_pipeline()

# ────────────────────────────────────────────
# STEP 2: TRAINING (semua varian)
# ────────────────────────────────────────────
from lstm_model import train_model, evaluate_model, compare_variants

VARIANTS = ["vanilla", "stacked", "bidirectional"]

for i, variant in enumerate(VARIANTS, 1):
    print(f"\n[STEP 2/{len(VARIANTS)*2}] TRAINING — {variant.upper()}")
    print("─" * 40)
    train_model(variant=variant, epochs=80, batch_size=64)

# ────────────────────────────────────────────
# STEP 3: EVALUASI
# ────────────────────────────────────────────
all_metrics = []
for variant in VARIANTS:
    print(f"\n[STEP 3] EVALUASI — {variant.upper()}")
    print("─" * 40)
    m, _, _ = evaluate_model(variant=variant)
    all_metrics.append(m)

# ────────────────────────────────────────────
# STEP 4: PERBANDINGAN
# ────────────────────────────────────────────
print("\n[STEP 4] PERBANDINGAN VARIAN")
print("─" * 40)
compare_variants()

# ────────────────────────────────────────────
# RINGKASAN AKHIR
# ────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════╗
║  ✓ PIPELINE SELESAI                                  ║
╠══════════════════════════════════════════════════════╣""")

for m in all_metrics:
    print(f"║  {m['variant']:15s} │ Acc: {m['accuracy']:5.2f}%  F1: {m['f1_score']:5.2f}%      ║")

best = max(all_metrics, key=lambda x: x['accuracy'])
print(f"""╠══════════════════════════════════════════════════════╣
║  🏆 Model Terbaik : {best['variant']:14s}  Acc {best['accuracy']:.2f}%  ║
╠══════════════════════════════════════════════════════╣
║  Output tersimpan di folder: outputs/                ║
║  Jalankan dashboard: python app.py                   ║
║  Buka browser: http://localhost:5000                 ║
╚══════════════════════════════════════════════════════╝
""")