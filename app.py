"""
app.py
Dashboard Streamlit – LSTM Forex Signal Classifier
Skripsi: Implementasi LSTM untuk Klasifikasi Sinyal Trading Forex

Jalankan dengan:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, os, pickle
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Forex Signal Classifier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Metric cards */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
}
.metric-card .sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 4px;
}

/* Signal badge */
.signal-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.05em;
}
.signal-BUY   { background: #052e16; color: #4ade80; border: 1.5px solid #22c55e; }
.signal-SELL  { background: #2d0e0e; color: #f87171; border: 1.5px solid #ef4444; }
.signal-NO-ENTRY { background: #1c1408; color: #fbbf24; border: 1.5px solid #f59e0b; }

/* Section header */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}

/* Predict card */
.predict-main {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 32px;
}

/* Staleness warning */
.stale-warning {
    background: #2d1a00;
    border: 1px solid #92400e;
    border-radius: 8px;
    padding: 10px 16px;
    color: #fbbf24;
    font-size: 0.82rem;
}
.fresh-badge {
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: 10px 16px;
    color: #4ade80;
    font-size: 0.82rem;
}

/* Proba bar */
.proba-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.proba-label { width: 80px; color: #94a3b8; }
.proba-bar-bg {
    flex: 1;
    background: #1e293b;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.proba-fill { height: 100%; border-radius: 4px; }
.proba-val { width: 52px; text-align: right; color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"

LABEL_NAMES = {0: "SELL", 1: "NO-ENTRY", 2: "BUY"}
SIGNAL_COLORS = {"BUY": "#22c55e", "SELL": "#ef4444", "NO-ENTRY": "#f59e0b"}


# ─── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_merged_df():
    path = f"{DATA_DIR}/dataset_merged.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_metrics(variant="stacked"):
    path = f"{OUTPUT_DIR}/metrics_{variant}.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=300, show_spinner=False)
def load_history(variant="stacked"):
    path = f"{MODEL_DIR}/history_{variant}.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=300, show_spinner=False)
def load_all_metrics():
    results = []
    for v in ["vanilla", "stacked", "bidirectional"]:
        mfile = f"{OUTPUT_DIR}/metrics_{v}.json"
        if os.path.exists(mfile):
            with open(mfile) as f:
                results.append(json.load(f))
    return results


def check_status():
    return {
        "trained":    os.path.exists(f"{MODEL_DIR}/lstm_stacked.keras"),
        "data_ready": os.path.exists(f"{DATA_DIR}/dataset_merged.csv"),
    }


def get_latest_prediction(df, variant="stacked"):
    """Prediksi dari window terakhir dataset."""
    try:
        from lstm_model import predict_signal
        from data_pipeline import FEATURE_COLS, SEQUENCE_LEN
    except ImportError:
        return None

    if df is None or len(df) < SEQUENCE_LEN:
        return None

    scaler_path = f"{DATA_DIR}/scaler.pkl"
    if not os.path.exists(scaler_path):
        return None

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    window_raw    = df[FEATURE_COLS].values[-SEQUENCE_LEN:]
    window_scaled = scaler.transform(window_raw)
    result        = predict_signal(window_scaled, variant)

    last_date  = df.index[-1]
    days_behind = (datetime.now() - last_date).days
    result["datetime"]    = str(last_date)
    result["close"]       = round(float(df["close"].iloc[-1]), 5)
    result["ma200"]       = round(float(df["ma200"].iloc[-1]), 5)
    result["days_behind"] = days_behind
    result["is_fresh"]    = days_behind <= 1
    return result


def run_pipeline_background():
    import subprocess, sys
    subprocess.Popen([sys.executable, "run_all.py"], cwd=os.getcwd())


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## LSTM Forex")
    st.markdown("**Klasifikasi Sinyal EUR/USD**")
    st.markdown("---")

    variant = st.selectbox(
        "Model Variant",
        options=["stacked", "vanilla", "bidirectional"],
        format_func=lambda x: {
            "stacked": "Stacked LSTM (default)",
            "vanilla": "Vanilla LSTM",
            "bidirectional": "Bidirectional LSTM",
        }[x],
    )

    st.markdown("---")
    status = check_status()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Data**  \n{'✅ Siap' if status['data_ready'] else '❌ Belum'}")
    with col2:
        st.markdown(f"**Model**  \n{'✅ Siap' if status['trained'] else '❌ Belum'}")

    st.markdown("---")

    if st.button("▶ Jalankan Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Menjalankan pipeline di background..."):
            run_pipeline_background()
        st.success("Pipeline dimulai. Refresh setelah beberapa menit.")

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#475569;'>Skripsi — LSTM Forex Signal Classification<br>EUR/USD Multi-Timeframe (H4 + D)</div>",
        unsafe_allow_html=True,
    )


# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_price, tab_training, tab_eval, tab_compare = st.tabs([
    "Prediksi", "Chart Harga", "Training", "Evaluasi", "Perbandingan"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Sinyal Trading Terkini")
    st.markdown(
        "<div class='section-header'>Prediksi dari window 30 candle H4 terakhir menggunakan model LSTM yang telah dilatih</div>",
        unsafe_allow_html=True,
    )

    df = load_merged_df()

    if not status["trained"] or not status["data_ready"]:
        st.warning("⚠️ Model atau dataset belum tersedia. Jalankan pipeline terlebih dahulu dari sidebar.")
    else:
        with st.spinner("Memuat prediksi..."):
            pred = get_latest_prediction(df, variant)

        if pred is None:
            st.error("Gagal menghasilkan prediksi. Pastikan model dan data tersedia.")
        else:
            sig = pred["signal"]

            # Freshness indicator
            if pred["is_fresh"]:
                st.markdown(
                    f"<div class='fresh-badge'>✅ Data terkini — {pred['datetime'][:10]}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='stale-warning'>⚠️ Data sudah {pred['days_behind']} hari di belakang pasar. "
                    f"Terakhir update: {pred['datetime'][:10]}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Main prediction block
            c1, c2, c3 = st.columns([1.2, 1, 1])

            with c1:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='label'>Sinyal</div>
                        <div style='margin: 12px 0;'>
                            <span class='signal-badge signal-{sig}'>{sig}</span>
                        </div>
                        <div class='sub'>Confidence: <b>{pred['confidence']}%</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='label'>Harga Close</div>
                        <div class='value'>{pred['close']}</div>
                        <div class='sub'>EUR/USD (H4 terakhir)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                delta = round(pred["close"] - pred["ma200"], 5)
                delta_pct = round((pred["close"] / pred["ma200"] - 1) * 100, 3)
                pos = "DI ATAS" if delta > 0 else "DI BAWAH"
                color_ma = "#4ade80" if delta > 0 else "#f87171"
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='label'>MA 200</div>
                        <div class='value' style='color:{color_ma};font-size:1.5rem;'>{pred['ma200']}</div>
                        <div class='sub'>Harga {pos} MA200 ({delta_pct:+.3f}%)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Probability bars
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Distribusi Probabilitas")

            probs = pred["probabilities"]
            bar_colors = {"BUY": "#22c55e", "SELL": "#ef4444", "NO-ENTRY": "#f59e0b"}

            for label, prob in probs.items():
                color = bar_colors.get(label, "#64748b")
                st.markdown(
                    f"""
                    <div class='proba-row'>
                        <div class='proba-label'>{label}</div>
                        <div class='proba-bar-bg'>
                            <div class='proba-fill' style='width:{prob}%;background:{color};'></div>
                        </div>
                        <div class='proba-val'>{prob:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred["confidence"],
                title={"text": "Confidence Level", "font": {"size": 14, "color": "#94a3b8"}},
                number={"suffix": "%", "font": {"size": 28, "color": "#f1f5f9"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar": {"color": SIGNAL_COLORS.get(sig, "#6366f1"), "thickness": 0.3},
                    "bgcolor": "#1e293b",
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, 40],  "color": "#1e293b"},
                        {"range": [40, 70], "color": "#1c2840"},
                        {"range": [70, 100],"color": "#132036"},
                    ],
                    "threshold": {
                        "line": {"color": SIGNAL_COLORS.get(sig, "#6366f1"), "width": 3},
                        "thickness": 0.75,
                        "value": pred["confidence"],
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                height=240,
                margin=dict(t=40, b=20, l=30, r=30),
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHART HARGA
# ══════════════════════════════════════════════════════════════════════════════
with tab_price:
    st.markdown("### Chart Harga EUR/USD + Sinyal")

    df = load_merged_df()
    if df is None:
        st.warning("Dataset belum tersedia.")
    else:
        n_candles = st.slider("Jumlah candle", 50, 500, 200, step=50)
        df_tail = df.tail(n_candles).copy()
        label_map = {0: "SELL", 1: "NO-ENTRY", 2: "BUY"}
        df_tail["signal_label"] = df_tail["signal"].map(lambda x: label_map.get(int(x), "NO-ENTRY"))

        # Pisahkan per sinyal untuk scatter
        df_buy  = df_tail[df_tail["signal_label"] == "BUY"]
        df_sell = df_tail[df_tail["signal_label"] == "SELL"]
        df_ne   = df_tail[df_tail["signal_label"] == "NO-ENTRY"]

        fig = go.Figure()

        # Close price line
        fig.add_trace(go.Scatter(
            x=df_tail.index, y=df_tail["close"],
            name="Close",
            line=dict(color="#6366f1", width=1.5),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.5f}<extra></extra>",
        ))

        # MA200 line
        fig.add_trace(go.Scatter(
            x=df_tail.index, y=df_tail["ma200"],
            name="MA 200",
            line=dict(color="#f59e0b", width=1, dash="dot"),
            hovertemplate="MA200: %{y:.5f}<extra></extra>",
        ))

        # Signal markers
        for df_sig, color, symbol, name in [
            (df_buy,  "#22c55e", "triangle-up",   "BUY"),
            (df_sell, "#ef4444", "triangle-down",  "SELL"),
            (df_ne,   "#f59e0b", "circle",         "NO-ENTRY"),
        ]:
            if len(df_sig):
                fig.add_trace(go.Scatter(
                    x=df_sig.index, y=df_sig["close"],
                    mode="markers",
                    name=name,
                    marker=dict(symbol=symbol, size=7, color=color, opacity=0.85),
                    hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.5f}}<extra></extra>",
                ))

        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8", family="Inter"),
            xaxis=dict(gridcolor="#1e293b", showgrid=True, zeroline=False),
            yaxis=dict(gridcolor="#1e293b", showgrid=True, zeroline=False,
                       tickformat=".5f"),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
            height=480,
            margin=dict(t=30, b=30, l=10, r=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Signal distribution donut
        st.markdown("#### Distribusi Sinyal")
        dist = df_tail["signal_label"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=dist.index.tolist(),
            values=dist.values.tolist(),
            hole=0.55,
            marker=dict(colors=[SIGNAL_COLORS.get(l, "#6366f1") for l in dist.index]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} candle (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
            showlegend=False,
            height=300,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAINING HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_training:
    st.markdown(f"### Training History — {variant.capitalize()}")

    h = load_history(variant)
    if not h:
        st.warning("History training belum tersedia. Latih model terlebih dahulu.")
    else:
        epochs = list(range(1, len(h.get("loss", [])) + 1))

        fig_train = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss (Train vs Val)", "Accuracy (Train vs Val)"),
        )

        fig_train.add_trace(go.Scatter(
            x=epochs, y=h.get("loss", []),
            name="Train Loss", line=dict(color="#6366f1", width=2),
        ), row=1, col=1)
        fig_train.add_trace(go.Scatter(
            x=epochs, y=h.get("val_loss", []),
            name="Val Loss", line=dict(color="#f59e0b", width=2, dash="dash"),
        ), row=1, col=1)

        train_acc = [v * 100 for v in h.get("accuracy", [])]
        val_acc   = [v * 100 for v in h.get("val_accuracy", [])]
        fig_train.add_trace(go.Scatter(
            x=epochs, y=train_acc,
            name="Train Acc", line=dict(color="#22c55e", width=2),
        ), row=1, col=2)
        fig_train.add_trace(go.Scatter(
            x=epochs, y=val_acc,
            name="Val Acc", line=dict(color="#ef4444", width=2, dash="dash"),
        ), row=1, col=2)

        fig_train.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
            height=380,
            margin=dict(t=50, b=30, l=10, r=10),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
        )
        for i in [1, 2]:
            fig_train.update_xaxes(gridcolor="#1e293b", title_text="Epoch", row=1, col=i)
            fig_train.update_yaxes(gridcolor="#1e293b", row=1, col=i)

        st.plotly_chart(fig_train, use_container_width=True)

        # Summary stat
        best_val_acc = max(val_acc) if val_acc else 0
        best_epoch   = val_acc.index(best_val_acc) + 1 if val_acc else "-"
        final_loss   = h.get("val_loss", [0])[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Best Val Accuracy", f"{best_val_acc:.2f}%")
        c2.metric("Best Epoch", best_epoch)
        c3.metric("Final Val Loss", f"{final_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUASI
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown(f"### Evaluasi Model — {variant.capitalize()}")

    m = load_metrics(variant)
    if not m:
        st.warning("Metrics belum tersedia. Jalankan evaluasi terlebih dahulu.")
    else:
        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        for col, key, label, color in [
            (c1, "accuracy",  "Accuracy",  "#6366f1"),
            (c2, "precision", "Precision", "#22c55e"),
            (c3, "recall",    "Recall",    "#f59e0b"),
            (c4, "f1_score",  "F1-Score",  "#ef4444"),
        ]:
            val = m.get(key, 0)
            col.markdown(
                f"""
                <div class='metric-card'>
                    <div class='label'>{label}</div>
                    <div class='value' style='color:{color};'>{val:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Radar chart
        categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values_    = [m.get("accuracy", 0), m.get("precision", 0),
                      m.get("recall", 0),   m.get("f1_score", 0)]
        values_.append(values_[0])  # close loop

        fig_radar = go.Figure(go.Scatterpolar(
            r=values_,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(99,102,241,0.2)",
            line=dict(color="#6366f1", width=2),
            name=variant.capitalize(),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1e293b",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#334155", color="#475569"),
                angularaxis=dict(gridcolor="#334155", color="#94a3b8"),
            ),
            paper_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
            height=340,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Confusion matrix dari file png (fallback teks)
        cm_path = f"{OUTPUT_DIR}/confusion_{variant}.png"
        if os.path.exists(cm_path):
            st.markdown("#### Confusion Matrix")
            st.image(cm_path, use_container_width=False, width=500)
        else:
            st.info("Confusion matrix belum tersedia. Jalankan `evaluate_model()` untuk menghasilkannya.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PERBANDINGAN MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### Perbandingan Varian Model LSTM")

    all_metrics = load_all_metrics()
    if not all_metrics:
        st.warning("Belum ada model yang dievaluasi. Latih dan evaluasi semua variant terlebih dahulu.")
    else:
        df_cmp = pd.DataFrame(all_metrics).set_index("variant")

        # Bar chart grouped
        metric_keys   = ["accuracy", "precision", "recall", "f1_score"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        bar_colors    = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444"]

        fig_cmp = go.Figure()
        for key, label, color in zip(metric_keys, metric_labels, bar_colors):
            fig_cmp.add_trace(go.Bar(
                name=label,
                x=[v.capitalize() for v in df_cmp.index],
                y=df_cmp[key].tolist(),
                marker_color=color,
                opacity=0.85,
                text=[f"{v:.1f}%" for v in df_cmp[key]],
                textposition="outside",
                textfont=dict(size=11),
            ))

        fig_cmp.update_layout(
            barmode="group",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0, 110], title="Nilai (%)"),
            legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1,
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
            margin=dict(t=50, b=30, l=10, r=10),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Highlight best
        best_model = df_cmp["f1_score"].idxmax()
        best_f1    = df_cmp["f1_score"].max()
        st.success(f"Model terbaik berdasarkan F1-Score: **{best_model.capitalize()}** ({best_f1:.2f}%)")

        # Table
        st.markdown("#### Tabel Perbandingan Lengkap")
        display_df = df_cmp[metric_keys].copy()
        display_df.columns = metric_labels
        display_df.index   = [v.capitalize() for v in display_df.index]
        display_df = display_df.map(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True)