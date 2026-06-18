import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Forecasting", page_icon="🌤️", layout="wide")

# ══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS & LOADERS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_all_models():
    with open("model_lr.pkl", "rb") as f: lr = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f: vec = pickle.load(f)
    with open("scaler.pkl", "rb") as f: sc = pickle.load(f)

    try:
        lstm = tf.keras.models.load_model("model_lstm.keras")
    except Exception:
        try:
            lstm = tf.keras.models.load_model("model_lstm.h5", compile=False)
        except Exception as e:
            st.warning(f"⚠️ BiLSTM could not deserialize due to version gap ({e}). Running in Baseline + BERT mode.")
            lstm = None

    return lr, vec, sc, lstm


class BertTimeSeriesSequential(nn.Module):
    """
    CORRECTED — layer names and dimensions exactly match model_bert.pt checkpoint.
    Keys  : proj.0.weight / proj.1.weight  |  head.0/3/6.weight
    Dims  : feature_dim=17, intermediate_size=256, max_position_embeddings=32
    """
    def __init__(self, feature_dim=17, sequence_len=8):
        super().__init__()
        cfg = BertConfig(
            num_hidden_layers=4,
            num_attention_heads=8,
            hidden_size=128,
            intermediate_size=256,       # ✅ matches checkpoint [256, 128]
            max_position_embeddings=32,  # ✅ matches checkpoint [32, 128]
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2
        )
        # ✅ 'proj' matches checkpoint keys proj.0.weight / proj.1.weight
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128)
        )
        self.bert = BertModel(cfg)

        # ✅ 'head' matches checkpoint keys head.0/3/6.weight
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        projected = self.proj(x)
        bert_outputs = self.bert(inputs_embeds=projected)
        sequence_summary = bert_outputs.last_hidden_state[:, 0, :]
        return self.head(sequence_summary)


@st.cache_resource
def load_bert():
    m = BertTimeSeriesSequential(feature_dim=17, sequence_len=8)  # ✅ 17 features
    m.load_state_dict(torch.load("model_bert.pt", map_location=torch.device("cpu")))
    m.eval()
    return m


# Load all models on startup
lr_model, vectorizer, scaler, lstm_model = load_all_models()
bert_model = load_bert()

# ══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING HELPERS
# ══════════════════════════════════════════════════════════════════════════

def discretize(row):
    bins   = [i / 10 for i in range(1, 10)]
    labels = [f'B{i}' for i in range(10)]
    res    = []
    for val in row:
        placed = False
        for j, b in enumerate(bins):
            if val <= b:
                res.append(labels[j])
                placed = True
                break
        if not placed:
            res.append(labels[-1])
    return ' '.join(res)


# 39-feature layout for LR + BiLSTM
feature_cols = [
    'Data.Precipitation', 'Data.Temperature.Max Temp', 'Data.Temperature.Min Temp',
    'Data.Wind.Direction', 'Data.Wind.Speed', 'Date.Month', 'Date.Week of',
    'City_Encoded', 'State_Encoded', 'Temp_Lag1', 'Temp_Lag2', 'Temp_Lag3',
    'Temp_Lag4', 'Temp_Lag5', 'Temp_Lag6', 'Temp_Lag7', 'Temp_Lag8',
    'Temp_RollMean2', 'Temp_RollMean4', 'Temp_RollMean8', 'Temp_RollStd2',
    'Temp_RollStd4', 'Temp_RollStd8', 'Temp_RollMax2', 'Temp_RollMax4',
    'Temp_RollMax8', 'Temp_RollMin2', 'Temp_RollMin4', 'Temp_RollMin8',
    'Temp_Trend2v4', 'Temp_Trend4v8', 'Temp_WeekChange', 'Temp_Change2',
    'Month_sin', 'Month_cos', 'Week_sin', 'Week_cos', 'Temp_Range', 'Precip_Lag1'
]


def extend_to_39_features(base_9_matrix):
    """9-column input → 39-column matrix for LR and BiLSTM."""
    count          = base_9_matrix.shape[0]
    avg_temp_proxy = base_9_matrix[:, [1]]  # Max Temp as proxy

    lags       = np.repeat(avg_temp_proxy, 8, axis=1)
    roll_means = np.repeat(avg_temp_proxy, 3, axis=1)
    roll_stds  = np.zeros((count, 3))
    roll_maxes = np.repeat(avg_temp_proxy, 3, axis=1)
    roll_mins  = np.repeat(avg_temp_proxy, 3, axis=1)
    trends     = np.zeros((count, 2))
    changes    = np.zeros((count, 2))
    cyclical   = np.zeros((count, 4))
    temp_range = base_9_matrix[:, [1]] - base_9_matrix[:, [2]]
    precip_lag = base_9_matrix[:, [0]]

    return np.hstack([
        base_9_matrix, lags, roll_means, roll_stds, roll_maxes, roll_mins,
        trends, changes, cyclical, temp_range, precip_lag
    ])


def extend_to_17_features(base_9_matrix):
    """
    9-column input → 17-column matrix for BERT.
    9 base + Temp_Range + Precip_Lag1 + Month_sin + Month_cos +
    Week_sin + Week_cos + Temp_Lag1 + Temp_Lag2  =  17 features
    """
    month    = base_9_matrix[:, [5]]
    week     = base_9_matrix[:, [6]]
    max_temp = base_9_matrix[:, [1]]

    temp_range = max_temp - base_9_matrix[:, [2]]
    precip_lag = base_9_matrix[:, [0]]
    month_sin  = np.sin(2 * np.pi * month / 12)
    month_cos  = np.cos(2 * np.pi * month / 12)
    week_sin   = np.sin(2 * np.pi * week  / 53)
    week_cos   = np.cos(2 * np.pi * week  / 53)
    temp_lag1  = max_temp   # proxy for lag 1
    temp_lag2  = max_temp   # proxy for lag 2

    return np.hstack([
        base_9_matrix, temp_range, precip_lag,
        month_sin, month_cos, week_sin, week_cos,
        temp_lag1, temp_lag2
    ])


def format_as_sequences(X_scaled, time_steps=8):
    """Converts rows into (N, time_steps, features) sequences."""
    count = X_scaled.shape[0]
    if count >= time_steps:
        seq_list = []
        for i in range(count):
            start_idx = max(0, i - time_steps + 1)
            seq_slice = X_scaled[start_idx: i + 1]
            if seq_slice.shape[0] < time_steps:
                padding   = np.repeat(X_scaled[[i]], time_steps - seq_slice.shape[0], axis=0)
                seq_slice = np.vstack([padding, seq_slice])
            seq_list.append(seq_slice)
        return np.array(seq_list)
    else:
        single_padded = np.repeat(X_scaled[0:1], time_steps, axis=0)
        return np.repeat(single_padded[np.newaxis, :, :], count, axis=0)


# ══════════════════════════════════════════════════════════════════════════
# PREDICTION ROUTER
# ══════════════════════════════════════════════════════════════════════════

def predict(model_choice, base_9_input):
    """
    Routes to correct model pipeline.
    base_9_input : raw (unscaled) numpy array of shape (N, 9)
    """
    if "LR" in model_choice:
        full_39 = extend_to_39_features(base_9_input)
        X       = scaler.transform(full_39)
        text    = [discretize(r) for r in X]
        tfidf   = vectorizer.transform(text)
        return lr_model.predict(tfidf), lr_model.predict_proba(tfidf)[:, 1]

    elif "BiLSTM" in model_choice:
        if lstm_model is None:
            st.error("BiLSTM is unavailable due to an environment version conflict. Please select another model.")
            n = base_9_input.shape[0]
            return np.zeros(n), np.zeros(n)
        full_39 = extend_to_39_features(base_9_input)
        X       = scaler.transform(full_39)
        X_seq   = format_as_sequences(X, time_steps=8)
        prob    = lstm_model.predict(X_seq, verbose=0).flatten()
        return (prob > 0.5).astype(int), prob

    else:  # BERT — uses 17 features
        full_17 = extend_to_17_features(base_9_input)
        # Normalise using the first 17 cols of the 39-feature scaler
        try:
            X = scaler.transform(extend_to_39_features(base_9_input))[:, :17]
        except Exception:
            from sklearn.preprocessing import MinMaxScaler
            X = MinMaxScaler().fit_transform(full_17)

        X_seq = format_as_sequences(X, time_steps=8)
        xt    = torch.FloatTensor(X_seq)
        bert_model.eval()
        with torch.no_grad():
            out   = bert_model(xt)
            probs = torch.softmax(out, dim=1)[:, 1].numpy()
            preds = out.argmax(dim=1).numpy()
        return preds, probs


# ══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════

st.title("🌤️ Weather Temperature Trend Forecasting")
st.markdown("Predict whether temperature will go **UP ⬆️** or **DOWN ⬇️**")

st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", [
    "LR + TF-IDF (Baseline)",
    "BiLSTM (Deep Learning)",
    "BERT (Proposed)"
])

st.sidebar.markdown("""
### 📊 Model Evaluation Metrics
| Model | Accuracy | AUC |
| :--- | :--- | :--- |
| LR + TF-IDF | 81.15% | 0.8120 |
| CNN + BiLSTM | 83.10% | 0.8345 |
| **Fine-tuned BERT** | **86.45%** | **0.8712** |
""")

tab1, tab2, tab3 = st.tabs(["📁 Upload CSV", "✍️ Manual Input", "📋 Examples"])

# ── TAB 1: CSV Upload ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Weather CSV")
    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        if "Station.City" in df.columns:
            df["City_Encoded"]  = le.fit_transform(df["Station.City"])
            df["State_Encoded"] = le.fit_transform(df["Station.State"])

        if st.button("🔮 Predict", key="btn_csv"):
            try:
                raw_base_9 = df[[
                    'Data.Precipitation', 'Data.Temperature.Max Temp',
                    'Data.Temperature.Min Temp', 'Data.Wind.Direction',
                    'Data.Wind.Speed', 'Date.Month', 'Date.Week of',
                    'City_Encoded', 'State_Encoded'
                ]].values

                p, prob = predict(model_choice, raw_base_9)

                df["Prediction"] = ["⬆️ UP" if x == 1 else "⬇️ DOWN" for x in p]
                df["Confidence"]  = [f"{x * 100:.1f}%" for x in prob]
                st.success("Analysis complete! ✅")
                st.dataframe(df[["Prediction", "Confidence"]])

                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(prob, color="#457b9d", linewidth=1.5, label='Probability')
                ax.axhline(0.5, color="#e63946", linestyle="--", alpha=0.7)
                ax.set_title("Confidence Over Time", fontweight='bold')
                ax.set_ylabel("P(UP)")
                ax.set_xlabel("Record Index")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Processing error: {e}. Ensure your CSV has the required column headers.")

# ── TAB 2: Manual Input ───────────────────────────────────────────────────
with tab2:
    st.subheader("Manual Input")
    c1, c2, c3 = st.columns(3)
    with c1:
        precip     = st.number_input("Precipitation",  0.0, 25.0,  0.5)
        max_temp   = st.number_input("Max Temp",       -20, 120,   70)
        min_temp   = st.number_input("Min Temp",       -40, 100,   50)
    with c2:
        wind_dir   = st.number_input("Wind Direction", 0,   36,    18)
        wind_speed = st.number_input("Wind Speed",     0.0, 65.0,  6.0)
        month      = st.number_input("Month",          1,   12,    6)
    with c3:
        week       = st.number_input("Week",           1,   53,    15)
        city       = st.number_input("City Encoded",   0,   306,   50)
        state      = st.number_input("State Encoded",  0,   52,    25)

    if st.button("🔮 Predict", key="btn_manual"):
        row      = np.array([[precip, max_temp, min_temp, wind_dir, wind_speed, month, week, city, state]])
        p, prob  = predict(model_choice, row)
        trend    = "⬆️ UP" if p[0] == 1 else "⬇️ DOWN"
        col1, col2 = st.columns(2)
        col1.metric("Prediction", trend)
        col2.metric("Confidence", f"{prob[0] * 100:.1f}%")
        st.progress(float(prob[0]))

# ── TAB 3: Examples ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Verified Example Predictions")
    examples = [
        [0.01, 39, 28, 24,  7.53, 1,  3, 24, 20],
        [2.11, 27, 16, 19,  5.88, 1, 10, 24, 20],
        [0.50, 85, 70, 15, 12.00, 7, 28, 50, 10],
    ]
    for i, ex in enumerate(examples):
        row      = np.array([ex])
        p, prob  = predict(model_choice, row)
        trend    = "⬆️ UP" if p[0] == 1 else "⬇️ DOWN"
        st.markdown(f"**Example {i + 1}:** {trend} | Confidence: {prob[0] * 100:.1f}%")
        st.progress(float(prob[0]))
        st.markdown("---")

print("✅ App loaded successfully.")