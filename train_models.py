import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and Preprocess Data
# ══════════════════════════════════════════════════════════════════════════
print("Loading and preprocessing data...")

df = pd.read_csv("weather (1).csv")

le_city  = LabelEncoder()
le_state = LabelEncoder()
df['City_Encoded']  = le_city.fit_transform(df['Station.City'])
df['State_Encoded'] = le_state.fit_transform(df['Station.State'])

df['Date.Full'] = pd.to_datetime(df['Date.Full'])
df = df.sort_values(['Station.City', 'Date.Full']).reset_index(drop=True)

# Lag features
df['Temp_Lag1'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].shift(1)
df['Temp_Lag2'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].shift(2)
df['Temp_Lag3'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].shift(3)
df['Temp_Lag4'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].shift(4)
df['Temp_RollMean4'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].transform(
    lambda x: x.rolling(4, min_periods=1).mean())
df['Temp_RollStd4'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].transform(
    lambda x: x.rolling(4, min_periods=1).std().fillna(0))

df.dropna(inplace=True)
df = df.reset_index(drop=True)

y_full  = df['Data.Temperature.Avg Temp']
y_label = (y_full.diff().shift(-1) > 0).astype(int)
df      = df[:-1].reset_index(drop=True)
y_label = y_label[:-1].reset_index(drop=True)

feature_cols = [
    'Data.Precipitation',
    'Data.Temperature.Max Temp',
    'Data.Temperature.Min Temp',
    'Data.Wind.Direction',
    'Data.Wind.Speed',
    'Date.Month',
    'Date.Week of',
    'City_Encoded',
    'State_Encoded',
    'Temp_Lag1',
    'Temp_Lag2',
    'Temp_Lag3',
    'Temp_Lag4',
    'Temp_RollMean4',
    'Temp_RollStd4'
]

NUM_FEATURES = len(feature_cols)
X = df[feature_cols].values
y = y_label.values

scaler  = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

total  = len(X_scaled)
split1 = int(total * 0.70)
split2 = int(total * 0.85)

X_train = X_scaled[:split1];  y_train = y[:split1]
X_val   = X_scaled[split1:split2]; y_val = y[split1:split2]
X_test  = X_scaled[split2:];  y_test  = y[split2:]

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Model 1: Logistic Regression + TF-IDF
# ══════════════════════════════════════════════════════════════════════════
print("\nTraining Model 1 — LR + TF-IDF...")

def discretize(row):
    labels = []
    for val in row:
        if   val <= 0.2: labels.append('VL')
        elif val <= 0.4: labels.append('L')
        elif val <= 0.6: labels.append('M')
        elif val <= 0.8: labels.append('H')
        else:            labels.append('VH')
    return ' '.join(labels)

X_train_text = [discretize(r) for r in X_train]
X_val_text   = [discretize(r) for r in X_val]
X_test_text  = [discretize(r) for r in X_test]

vectorizer    = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf   = vectorizer.transform(X_val_text)
X_test_tfidf  = vectorizer.transform(X_test_text)

model_lr = LogisticRegression(C=1, max_iter=1000, random_state=42)
model_lr.fit(X_train_tfidf, y_train)

with open('model_lr.pkl',   'wb') as f: pickle.dump(model_lr,    f)
with open('vectorizer.pkl', 'wb') as f: pickle.dump(vectorizer, f)
print("LR model saved!")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Model 2: BiLSTM
# ══════════════════════════════════════════════════════════════════════════
print("\nTraining Model 2 — BiLSTM...")

X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_lstm   = X_val.reshape(X_val.shape[0],   1, X_val.shape[1])
X_test_lstm  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

model_lstm = Sequential([
    Bidirectional(LSTM(64, return_sequences=True),
                  input_shape=(1, NUM_FEATURES)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1,  activation='sigmoid')
])

model_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5,
                            restore_best_weights=True)

lstm_history = model_lstm.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=50, batch_size=32,
    callbacks=[early_stop], verbose=1
)

# FIXED: Saved model as native .keras file format instead of legacy .h5 format
model_lstm.save('model_lstm.keras')
print("BiLSTM model saved!")

# Plot BiLSTM curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(lstm_history.history['loss'],     label='Train Loss')
ax1.plot(lstm_history.history['val_loss'], label='Val Loss')
ax1.set_title('BiLSTM Loss Curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(lstm_history.history['accuracy'],     label='Train Accuracy')
ax2.plot(lstm_history.history['val_accuracy'], label='Val Accuracy')
ax2.set_title('BiLSTM Accuracy Curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.tight_layout()
plt.savefig('bilstm_curves.png', dpi=150)
plt.show()
print("BiLSTM curves saved as bilstm_curves.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Model 3: Fine-tuned BERT
# ══════════════════════════════════════════════════════════════════════════
print("\nTraining Model 3 — BERT...")

class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        # FIXED: Explicitly copies array to avoid PyTorch read-only array warnings
        self.y = torch.LongTensor(y.copy())
    def __len__(self):       return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class BertForTimeSeries(nn.Module):
    def __init__(self, input_features=15):  # Dynamic initialization layer
        super().__init__()
        config = BertConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=64,
            intermediate_size=128,
            max_position_embeddings=16,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.input_proj = nn.Linear(input_features, 64)
        self.bert       = BertModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        x       = x.unsqueeze(1)
        x       = self.input_proj(x)
        outputs = self.bert(inputs_embeds=x)
        cls     = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls)

device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_loader = DataLoader(WeatherDataset(X_train, y_train), batch_size=32, shuffle=False)
val_loader   = DataLoader(WeatherDataset(X_val,   y_val),   batch_size=32, shuffle=False)

model_bert = BertForTimeSeries(input_features=NUM_FEATURES).to(device)
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model_bert.parameters(), lr=2e-4)

# Track curves for BERT
bert_train_losses, bert_val_losses = [], []
bert_train_accs,   bert_val_accs   = [], []
best_val_loss = float('inf')

for epoch in range(20):
    # Training
    model_bert.train()
    t_loss, t_correct, t_total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model_bert(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item()
        t_correct += (out.argmax(1) == yb).sum().item()
        t_total   += yb.size(0)

    # Validation
    model_bert.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out    = model_bert(xb)
            loss   = criterion(out, yb)
            v_loss    += loss.item()
            v_correct += (out.argmax(1) == yb).sum().item()
            v_total   += yb.size(0)

    avg_tl = t_loss / len(train_loader)
    avg_vl = v_loss / len(val_loader)
    avg_ta = t_correct / t_total
    avg_va = v_correct / v_total

    bert_train_losses.append(avg_tl)
    bert_val_losses.append(avg_vl)
    bert_train_accs.append(avg_ta)
    bert_val_accs.append(avg_va)

    if avg_vl < best_val_loss:
        best_val_loss = avg_vl
        torch.save(model_bert.state_dict(), 'model_bert.pt')

    print(f"Epoch {epoch+1:02d}/20 | "
          f"Train Loss: {avg_tl:.4f} Acc: {avg_ta:.4f} | "
          f"Val Loss: {avg_vl:.4f} Acc: {avg_va:.4f}")

print("BERT model saved!")

# Plot BERT curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(bert_train_losses, label='Train Loss')
ax1.plot(bert_val_losses,   label='Val Loss')
ax1.set_title('BERT Loss Curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(bert_train_accs, label='Train Accuracy')
ax2.plot(bert_val_accs,   label='Val Accuracy')
ax2.set_title('BERT Accuracy Curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.tight_layout()
plt.savefig('bert_curves.png', dpi=150)
plt.show()
print("BERT curves saved as bert_curves.png")

print("\nAll models trained and saved!")
print("Run evaluate.py next.");