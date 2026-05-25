
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Forecasting",
                   page_icon="🌤️", layout="wide")

@st.cache_resource
def load_all_models():
    with open("model_lr.pkl",   "rb") as f: lr  = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f: vec = pickle.load(f)
    with open("scaler.pkl",     "rb") as f: sc  = pickle.load(f)
    lstm = load_model("model_lstm.h5")
    return lr, vec, sc, lstm

class BertForTimeSeries(nn.Module):
    def __init__(self, input_features=9):
        super().__init__()
        from transformers import BertConfig, BertModel
        config = BertConfig(
            num_hidden_layers=2, num_attention_heads=4,
            hidden_size=64, intermediate_size=128,
            max_position_embeddings=16,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.input_proj = nn.Linear(input_features, 64)
        self.bert = BertModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 2)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        outputs = self.bert(inputs_embeds=x)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

@st.cache_resource
def load_bert():
    m = BertForTimeSeries()
    m.load_state_dict(torch.load("model_bert.pt",
                      map_location=torch.device("cpu")))
    m.eval()
    return m

lr_model, vectorizer, scaler, lstm_model = load_all_models()
bert_model = load_bert()

def discretize(row):
    labels = []
    for val in row:
        if val <= 0.2:   labels.append("VL")
        elif val <= 0.4: labels.append("L")
        elif val <= 0.6: labels.append("M")
        elif val <= 0.8: labels.append("H")
        else:            labels.append("VH")
    return " ".join(labels)

feature_cols = [
    "Data.Precipitation","Data.Temperature.Max Temp",
    "Data.Temperature.Min Temp","Data.Wind.Direction",
    "Data.Wind.Speed","Date.Month","Date.Week of",
    "City_Encoded","State_Encoded"
]

def predict(model_choice, X_scaled):
    if "LR" in model_choice:
        text  = [discretize(r) for r in X_scaled]
        tfidf = vectorizer.transform(text)
        return lr_model.predict(tfidf), lr_model.predict_proba(tfidf)[:,1]
    elif "BiLSTM" in model_choice:
        X3d  = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        prob = lstm_model.predict(X3d).flatten()
        return (prob > 0.5).astype(int), prob
    else:
        xt = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            out   = bert_model(xt)
            probs = torch.softmax(out, dim=1)[:,1].numpy()
            preds = out.argmax(dim=1).numpy()
        return preds, probs

st.title("🌤️ Weather Temperature Trend Forecasting")
st.markdown("Predict whether temperature will go **UP ⬆️** or **DOWN ⬇️**")

st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", [
    "LR + TF-IDF (Baseline)",
    "BiLSTM (Deep Learning)",
    "BERT (Proposed)"
])
st.sidebar.markdown("""
### 📊 Model AUC Scores
| Model | AUC |
|---|---|
| LR + TF-IDF | 0.52 |
| BiLSTM | 0.63 |
| **BERT** | **0.69** |
""")

tab1, tab2, tab3 = st.tabs(["📁 Upload CSV","✍️ Manual Input","📋 Examples"])

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
        if st.button("🔮 Predict"):
            X  = scaler.transform(df[feature_cols].values)
            p, prob = predict(model_choice, X)
            df["Prediction"] = ["⬆️ UP" if x==1 else "⬇️ DOWN" for x in p]
            df["Confidence"] = [f"{x*100:.1f}%" for x in prob]
            st.success("Done! ✅")
            st.dataframe(df[["Prediction","Confidence"]])
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(prob, color="blue", linewidth=1)
            ax.axhline(0.5, color="red", linestyle="--")
            ax.set_title("Confidence Over Time")
            st.pyplot(fig)

with tab2:
    st.subheader("Manual Input")
    c1, c2, c3 = st.columns(3)
    with c1:
        precip   = st.number_input("Precipitation",0.0,25.0,0.5)
        max_temp = st.number_input("Max Temp",-20,120,70)
        min_temp = st.number_input("Min Temp",-40,100,50)
    with c2:
        wind_dir   = st.number_input("Wind Direction",0,36,18)
        wind_speed = st.number_input("Wind Speed",0.0,65.0,6.0)
        month      = st.number_input("Month",1,12,6)
    with c3:
        week  = st.number_input("Week",1,31,15)
        city  = st.number_input("City Encoded",0,306,50)
        state = st.number_input("State Encoded",0,52,25)
    if st.button("🔮 Predict"):
        row = np.array([[precip,max_temp,min_temp,
                         wind_dir,wind_speed,month,
                         week,city,state]])
        X = scaler.transform(row)
        p, prob = predict(model_choice, X)
        trend = "⬆️ UP" if p[0]==1 else "⬇️ DOWN"
        col1, col2 = st.columns(2)
        col1.metric("Prediction", trend)
        col2.metric("Confidence", f"{prob[0]*100:.1f}%")
        st.progress(float(prob[0]))

with tab3:
    st.subheader("3 Example Predictions")
    examples = [
        [0.01,39,28,24,7.53,1, 3,24,20],
        [2.11,27,16,19,5.88,1,10,24,20],
        [0.50,85,70,15,12.0,7,28,50,10],
    ]
    for i, ex in enumerate(examples):
        X = scaler.transform(np.array([ex]))
        p, prob = predict(model_choice, X)
        trend = "⬆️ UP" if p[0]==1 else "⬇️ DOWN"
        st.markdown(f"**Example {i+1}:** {trend} | Confidence: {prob[0]*100:.1f}%")
        st.progress(float(prob[0]))
        st.markdown("---")
