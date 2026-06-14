"""
lstm_model.py
Arsitektur, Training, dan Evaluasi Model LSTM
Skripsi: Implementasi LSTM untuk Klasifikasi Sinyal Trading Forex
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, json
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Bidirectional, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR   = "data"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = {0: "SELL", 1: "NO-ENTRY", 2: "BUY"}
COLORS      = {"SELL": "#ef4444", "NO-ENTRY": "#f59e0b", "BUY": "#22c55e"}


# ─────────────────────────────────────────────
# 1. BANGUN ARSITEKTUR LSTM
# ─────────────────────────────────────────────
def build_model(seq_len, n_features, n_classes=3, variant="stacked"):
    """
    Variant:
        vanilla     – 1 LSTM layer
        stacked     – 2 LSTM layers bertumpuk (default skripsi)
        bidirectional – Bi-LSTM
    """
    model = Sequential(name=f"LSTM_{variant}")
    model.add(Input(shape=(seq_len, n_features)))

    if variant == "vanilla":
        model.add(LSTM(128, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

    elif variant == "stacked":
        model.add(LSTM(128, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

    elif variant == "bidirectional":
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────
# 2. TRAINING
# ─────────────────────────────────────────────
def train_model(variant="stacked", epochs=80, batch_size=64):
    print("=" * 55)
    print(f"  TRAINING MODEL LSTM – {variant.upper()}")
    print("=" * 55)

    # Load data
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")

    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    print(f"\n  Sequence length : {seq_len}")
    print(f"  Features        : {n_features}")
    print(f"  Train samples   : {len(X_train)}")
    print(f"  Val samples     : {len(X_val)}\n")

    # Class weights (atasi imbalance)
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    print(f"  Class weights   : { {LABEL_NAMES[k]: round(v,2) for k,v in class_weight.items()} }\n")

    model = build_model(seq_len, n_features, n_classes=3, variant=variant)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(f"{MODEL_DIR}/best_{variant}.keras", save_best_only=True, monitor="val_accuracy", verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Simpan model & history
    model.save(f"{MODEL_DIR}/lstm_{variant}.keras")
    with open(f"{MODEL_DIR}/history_{variant}.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    print(f"\n✓ Model disimpan: {MODEL_DIR}/lstm_{variant}.keras")
    _plot_training(history, variant)
    return model, history


# ─────────────────────────────────────────────
# 3. EVALUASI
# ─────────────────────────────────────────────
def evaluate_model(variant="stacked"):
    print("\n" + "=" * 55)
    print(f"  EVALUASI MODEL – {variant.upper()}")
    print("=" * 55)

    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    model = load_model(f"{MODEL_DIR}/lstm_{variant}.keras")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")

    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        zero_division=0
    ))

    metrics = {
        "variant": variant,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2)
    }
    with open(f"{OUTPUT_DIR}/metrics_{variant}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_confusion_matrix(y_test, y_pred, variant)
    return metrics, y_pred, y_pred_prob


# ─────────────────────────────────────────────
# 4. COMPARE VARIANTS
# ─────────────────────────────────────────────
def compare_variants():
    """Bandingkan Vanilla vs Stacked vs Bi-LSTM."""
    results = []
    for v in ["vanilla", "stacked", "bidirectional"]:
        mpath = f"{MODEL_DIR}/lstm_{v}.keras"
        mfile = f"{OUTPUT_DIR}/metrics_{v}.json"
        if os.path.exists(mpath) and os.path.exists(mfile):
            with open(mfile) as f:
                results.append(json.load(f))
    if not results:
        print("Tidak ada model untuk dibandingkan.")
        return

    df = pd.DataFrame(results).set_index("variant")
    print("\n" + "=" * 55)
    print("  PERBANDINGAN VARIAN MODEL")
    print("=" * 55)
    print(df.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.2
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    palette = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, palette)):
        ax.bar(x + i * width, df[metric], width, label=metric.capitalize(), color=color, alpha=0.85)

    ax.set_xlabel("Varian Model")
    ax.set_ylabel("Nilai (%)")
    ax.set_title("Perbandingan Performa Varian LSTM")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([v.capitalize() for v in df.index])
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison.png", dpi=150)
    plt.close()
    print(f"\n✓ Grafik perbandingan disimpan: {OUTPUT_DIR}/comparison.png")
    return df


# ─────────────────────────────────────────────
# 5. PLOT HELPERS
# ─────────────────────────────────────────────
def _plot_training(history, variant):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training History – LSTM {variant.capitalize()}", fontsize=13)

    axes[0].plot(history.history["loss"], label="Train Loss", color="#6366f1")
    axes[0].plot(history.history["val_loss"], label="Val Loss", color="#f59e0b")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Acc", color="#6366f1")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc", color="#f59e0b")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_{variant}.png", dpi=150)
    plt.close()
    print(f"✓ Plot training disimpan: {OUTPUT_DIR}/training_{variant}.png")


def _plot_confusion_matrix(y_true, y_pred, variant):
    cm = confusion_matrix(y_true, y_pred)
    labels = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="gray")
    plt.title(f"Confusion Matrix – LSTM {variant.capitalize()}")
    plt.ylabel("Aktual")
    plt.xlabel("Prediksi")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_{variant}.png", dpi=150)
    plt.close()
    print(f"✓ Confusion matrix disimpan: {OUTPUT_DIR}/confusion_{variant}.png")


# ─────────────────────────────────────────────
# 6. PREDICT REAL-TIME (Single Window)
# ─────────────────────────────────────────────
def predict_signal(window: np.ndarray, variant="stacked"):
    """
    Prediksi sinyal dari satu window candle.
    window: shape (seq_len, n_features), sudah dinormalisasi
    """
    model = load_model(f"{MODEL_DIR}/lstm_{variant}.keras")
    X = window[np.newaxis, ...]   # (1, seq_len, n_features)
    proba = model.predict(X, verbose=0)[0]
    pred  = int(np.argmax(proba))
    return {
        "signal": LABEL_NAMES[pred],
        "confidence": round(float(proba[pred]) * 100, 2),
        "probabilities": {
            LABEL_NAMES[i]: round(float(proba[i]) * 100, 2)
            for i in range(3)
        }
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    variant = sys.argv[1] if len(sys.argv) > 1 else "stacked"
    model, history = train_model(variant=variant)
    metrics, _, _  = evaluate_model(variant=variant)
    compare_variants()