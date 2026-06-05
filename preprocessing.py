import pandas as pd


import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── Load Data ─────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("weather (1).csv")

# ── Encode City and State ─────────────────────────────────────────────────
le_city  = LabelEncoder()
le_state = LabelEncoder()
df['City_Encoded']  = le_city.fit_transform(df['Station.City'])
df['State_Encoded'] = le_state.fit_transform(df['Station.State'])

# ── Fix Date Column ───────────────────────────────────────────────────────
df['Date.Full'] = pd.to_datetime(df['Date.Full'])
df = df.sort_values(['Station.City', 'Date.Full']).reset_index(drop=True)

# ── Add Lag Features (4 weeks history per city) ───────────────────────────
print("Adding lag features...")
df['Temp_Lag1'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].shift(1)
df['Temp_Lag2'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].shift(2)
df['Temp_Lag3'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].shift(3)
df['Temp_Lag4'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].shift(4)

# ── Add Rolling Statistics ────────────────────────────────────────────────
df['Temp_RollMean4'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].transform(
    lambda x: x.rolling(4, min_periods=1).mean()
)
df['Temp_RollStd4'] = df.groupby('Station.City')['Data.Temperature.Avg Temp'].transform(
    lambda x: x.rolling(4, min_periods=1).std().fillna(0)
)

# ── Drop rows with NaN from lag ───────────────────────────────────────────
df.dropna(inplace=True)
df = df.reset_index(drop=True)
print(f"Shape after adding lag features: {df.shape}")

# ── Create UP/DOWN Labels ─────────────────────────────────────────────────
y = df['Data.Temperature.Avg Temp']
y_label = (y.diff().shift(-1) > 0).astype(int)
df = df[:-1].reset_index(drop=True)
y_label = y_label[:-1].reset_index(drop=True)

# ── Feature Columns ───────────────────────────────────────────────────────
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

X = df[feature_cols].values
y = y_label.values

print(f"Features shape : {X.shape}")
print(f"Labels shape   : {y.shape}")
print(f"UP count       : {(y == 1).sum()}")
print(f"DOWN count     : {(y == 0).sum()}")

# ── Normalize ─────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── Temporal Split 70/15/15 ───────────────────────────────────────────────
total  = len(X_scaled)
split1 = int(total * 0.70)
split2 = int(total * 0.85)

X_train = X_scaled[:split1]
X_val   = X_scaled[split1:split2]
X_test  = X_scaled[split2:]
y_train = y[:split1]
y_val   = y[split1:split2]
y_test  = y[split2:]

print(f"\nTrain : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")

# ── Save Scaler ───────────────────────────────────────────────────────────
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nPreprocessing complete! scaler.pkl saved.")
print("Run train_models.py next.")