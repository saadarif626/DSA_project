import pandas as pd;
import numpy as np;
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
    classification_report, roc_curve
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Rebuild Data (same as train_models.py)
# ══════════════════════════════════════════════════════════════════════════
print("Rebuilding data splits...")

df = pd.read_csv("weather (1).csv")

le_city  = LabelEncoder()
le_state = LabelEncoder()
df['City_Encoded']  = le_city.fit_transform(df['Station.City'])
df['State_Encoded'] = le_state.fit_transform(df['Station.State'])

df['Date.Full'] = pd.to_datetime(df['Date.Full'])
df = df.sort_values(['Station.City', 'Date.Full']).reset_index(drop=True)

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

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

total  = len(X_scaled)
split1 = int(total * 0.70)
split2 = int(total * 0.85)

X_train = X_scaled[:split1];  y_train = y[:split1]
X_val   = X_scaled[split1:split2]; y_val = y[split1:split2]
X_test  = X_scaled[split2:];  y_test  = y[split2:]

print(f"Test set: {X_test.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Load All Models
# ══════════════════════════════════════════════════════════════════════════
print("\nLoading models...")

# LR
with open('model_lr.pkl',   'rb') as f: model_lr   = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)

# BiLSTM
from tensorflow.keras.models import load_model
model_lstm = load_model('model_lstm.h5')

# BERT
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):       return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class BertForTimeSeries(nn.Module):
    def __init__(self, input_features=15):
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
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.3),   nn.Linear(32, 2)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        return self.classifier(self.bert(inputs_embeds=x).last_hidden_state[:, 0, :])

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_bert = BertForTimeSeries(input_features=NUM_FEATURES).to(device)
model_bert.load_state_dict(torch.load('model_bert.pt', map_location=device))
model_bert.eval()
print("All models loaded!")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Generate Predictions
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating predictions...")

def discretize(row):
    labels = []
    for val in row:
        if   val <= 0.2: labels.append('VL')
        elif val <= 0.4: labels.append('L')
        elif val <= 0.6: labels.append('M')
        elif val <= 0.8: labels.append('H')
        else:            labels.append('VH')
    return ' '.join(labels)

# LR predictions
X_test_text  = [discretize(r) for r in X_test]
X_test_tfidf = vectorizer.transform(X_test_text)
y_pred_lr    = model_lr.predict(X_test_tfidf)
y_prob_lr    = model_lr.predict_proba(X_test_tfidf)[:, 1]

# BiLSTM predictions
X_test_lstm  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_prob_lstm  = model_lstm.predict(X_test_lstm, verbose=0).flatten()
y_pred_lstm  = (y_prob_lstm > 0.5).astype(int)

# BERT predictions
test_loader  = DataLoader(WeatherDataset(X_test, y_test), batch_size=32)
all_preds, all_probs, all_true = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb      = xb.to(device)
        out     = model_bert(xb)
        probs   = torch.softmax(out, dim=1)[:, 1]
        preds   = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_true.extend(yb.numpy())

y_pred_bert = np.array(all_preds)
y_prob_bert = np.array(all_probs)
y_true_bert = np.array(all_true)

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Overall Metrics Table
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("         OVERALL MODEL COMPARISON")
print("="*65)
print(f"{'Metric':<12} {'LR+TF-IDF':>12} {'BiLSTM':>10} {'BERT':>10} {'Winner':>10}")
print("-"*65)

metrics_data = [
    ('Accuracy',  accuracy_score(y_test, y_pred_lr),
                  accuracy_score(y_test, y_pred_lstm),
                  accuracy_score(y_true_bert, y_pred_bert)),
    ('Precision', precision_score(y_test, y_pred_lr),
                  precision_score(y_test, y_pred_lstm),
                  precision_score(y_true_bert, y_pred_bert)),
    ('Recall',    recall_score(y_test, y_pred_lr),
                  recall_score(y_test, y_pred_lstm),
                  recall_score(y_true_bert, y_pred_bert)),
    ('F1-Score',  f1_score(y_test, y_pred_lr),
                  f1_score(y_test, y_pred_lstm),
                  f1_score(y_true_bert, y_pred_bert)),
    ('ROC-AUC',   roc_auc_score(y_test, y_prob_lr),
                  roc_auc_score(y_test, y_prob_lstm),
                  roc_auc_score(y_true_bert, y_prob_bert)),
]

for metric, lr, lstm, bert in metrics_data:
    winner = ['LR', 'BiLSTM', 'BERT'][np.argmax([lr, lstm, bert])]
    print(f"{metric:<12} {lr:>12.4f} {lstm:>10.4f} {bert:>10.4f} {winner:>10}")

print("="*65)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Per-Class Metrics (Teacher's Requirement)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("    PER-CLASS METRICS — LR + TF-IDF")
print("="*65)
print(classification_report(y_test, y_pred_lr,
      target_names=['DOWN (0)', 'UP (1)'], digits=4))

print("="*65)
print("    PER-CLASS METRICS — BiLSTM")
print("="*65)
print(classification_report(y_test, y_pred_lstm,
      target_names=['DOWN (0)', 'UP (1)'], digits=4))

print("="*65)
print("    PER-CLASS METRICS — BERT")
print("="*65)
print(classification_report(y_true_bert, y_pred_bert,
      target_names=['DOWN (0)', 'UP (1)'], digits=4))

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Confusion Matrices Side by Side
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models_info = [
    ('LR + TF-IDF', y_test,      y_pred_lr,   'Blues'),
    ('BiLSTM',      y_test,      y_pred_lstm, 'Greens'),
    ('BERT',        y_true_bert, y_pred_bert, 'Oranges'),
]

for ax, (name, yt, yp, cmap) in zip(axes, models_info):
    cm   = confusion_matrix(yt, yp)
    disp = ConfusionMatrixDisplay(cm, display_labels=['DOWN', 'UP'])
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f'Confusion Matrix\n{name}')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
print("Saved: confusion_matrices.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — ROC Curves All Models
# ══════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 6))

fpr, tpr, _ = roc_curve(y_test,      y_prob_lr)
plt.plot(fpr, tpr, color='red',    label=f'LR + TF-IDF (AUC = {roc_auc_score(y_test, y_prob_lr):.2f})')

fpr, tpr, _ = roc_curve(y_test,      y_prob_lstm)
plt.plot(fpr, tpr, color='green',  label=f'BiLSTM (AUC = {roc_auc_score(y_test, y_prob_lstm):.2f})')

fpr, tpr, _ = roc_curve(y_true_bert, y_prob_bert)
plt.plot(fpr, tpr, color='blue',   label=f'BERT (AUC = {roc_auc_score(y_true_bert, y_prob_bert):.2f})')

plt.plot([0,1],[0,1], 'k--', label='Random (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()
print("Saved: roc_curves.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — Bar Chart Comparison
# ══════════════════════════════════════════════════════════════════════════
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
lr_scores    = [m[1] for m in metrics_data]
lstm_scores  = [m[2] for m in metrics_data]
bert_scores  = [m[3] for m in metrics_data]

x     = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 5))
b1 = ax.bar(x - width, lr_scores,   width, label='LR + TF-IDF', color='#FF6B6B', edgecolor='black')
b2 = ax.bar(x,         lstm_scores, width, label='BiLSTM',       color='#4ECDC4', edgecolor='black')
b3 = ax.bar(x + width, bert_scores, width, label='BERT',         color='#45B7D1', edgecolor='black')

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.legend()
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("Saved: model_comparison.png")

print("\nAll evaluation complete!")
print("Files saved:")
print("  confusion_matrices.png")
print("  roc_curves.png")
print("  model_comparison.png")