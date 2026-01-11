import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

BASE_PATH = "/Users/grzegorz/crypto_ml"

DATA_PATH = os.path.join(BASE_PATH, "btc.csv")
MODEL_PATH = os.path.join(BASE_PATH, "model.h5")

WINDOW = 30

print("📥 Loading data...")

df = pd.read_csv(DATA_PATH)

# Try to find price column automatically
if "price" in df.columns:
    prices = df[["price"]]
elif "Close" in df.columns:
    prices = df[["Close"]]
else:
    prices = df.iloc[:, -1:]  # last column fallback

values = prices.values

print("📊 Total samples:", len(values))

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# Create sequences
def make_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled, WINDOW)

print("🧠 Training samples:", X.shape)

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(WINDOW, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("🚀 Training model...")
model.fit(X, y, epochs=20, batch_size=32)

# Save model
model.save(MODEL_PATH)

print("✅ Model saved to:", MODEL_PATH)
print("📦 File created: model.h5")
