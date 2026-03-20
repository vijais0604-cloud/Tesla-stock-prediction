import joblib
import pandas as pd 
import numpy as np 
from  fastapi import FastAPI, UploadFile, File
from tensorflow import keras
from tensorflow.keras.models import load_model


model = load_model("best_lstm_model.h5")
features=joblib.load("features.pkl")
dataset_scaler=joblib.load("dataset_scaler.pkl")   
target_scaler=joblib.load("target_scaler.pkl")

def data_preprocess(data):
    df = pd.DataFrame(data)

    # Ensure datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # HL Range
    df["HL Range"] = df["High"] - df["Low"]

    # Rolling volume
    df["Volume-5"] = df["Volume"].rolling(5).mean()

    # Date features
    df["Month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek

    # Cyclical encoding
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)

    # Returns
    df["Returns"] = df["Adj Close"].pct_change()

    # Volatility
    df["Volatility_5"] = df["Returns"].rolling(5).std()
    df["Volatility_10"] = df["Returns"].rolling(10).std()

    # SMA
    df["SMA-10"] = df["Adj Close"].rolling(10).mean()

    # 🔥 Handle NaNs properly (instead of dropna)
    df = df.bfill().ffill()

    # Ensure correct feature order
    df = df.reindex(columns=features)

    # 🔥 FIX: enforce sequence length
    SEQ_LEN = 20

    if len(df) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} rows, got {len(df)}")

    # Take only last 20 timesteps
    df = df.tail(SEQ_LEN)

    # Scale
    scaled = dataset_scaler.transform(df)

    # Reshape for LSTM
    X = scaled.reshape(1, SEQ_LEN, len(features))

    return X

# ---------------- API ----------------
app = FastAPI()
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        X = data_preprocess(df)

        pred = model.predict(X)

        pred_actual = target_scaler.inverse_transform(pred)

        return {"predicted_adj_close": float(pred_actual[0][0])}

    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        traceback.print_exc()
        return {"error": str(e)}