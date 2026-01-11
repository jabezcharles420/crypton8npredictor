from fastapi import FastAPI, HTTPException
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import load_model

app = FastAPI(title="Crypto ML Prediction API")

BASE_PATH = os.getenv("DATA_PATH", ".")

DATA_PATH = os.path.join(BASE_PATH, "btc.csv")
MODEL_PATH = os.path.join(BASE_PATH, "model.h5")

WINDOW = 30

@app.get("/")
def root():
    return {"status": "ok", "message": "Crypto ML API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict():
    try:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=500, detail="btc.csv not found")

        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="model.h5 not found")

        # Load data
        df = pd.read_csv(DATA_PATH)

        # Auto-detect price column
        if "price" in df.columns:
            prices = df[["price"]]
        elif "Close" in df.columns:
            prices = df[["Close"]]
        else:
            prices = df.iloc[:, -1:]

        values = prices.values

        # Scale
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)

        # Make sequences
        X, y = [], []
        for i in range(len(scaled) - WINDOW):
            X.append(scaled[i:i+WINDOW])
            y.append(scaled[i+WINDOW])

        X = np.array(X)
        y = np.array(y)

        if len(X) < 10:
            raise HTTPException(status_code=400, detail="Not enough data points")

        # Load model
        model = load_model(MODEL_PATH, compile=False)


        # Evaluate
        pred = model.predict(X, verbose=0)

        r2 = float(r2_score(y, pred))
        mae = float(mean_absolute_error(y, pred))

        # Predict future 7 steps
        last_seq = X[-1]
        future = []

        for _ in range(7):
            p = model.predict(last_seq.reshape(1, WINDOW, 1), verbose=0)
            future.append(p[0][0])
            last_seq = np.append(last_seq[1:], p).reshape(WINDOW, 1)

        future_prices = scaler.inverse_transform(np.array(future).reshape(-1,1))

        return {
            "r2_score": r2,
            "mae": mae,
            "future_prediction": future_prices.flatten().tolist(),
            "points_used": int(len(X))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
