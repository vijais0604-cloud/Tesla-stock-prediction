📈 Stock Price Prediction using LSTM & FastAPI

🚀 Overview

This project focuses on predicting stock prices using a Long Short-Term Memory (LSTM) deep learning model. It includes an end-to-end pipeline from data preprocessing and feature engineering to model training, evaluation, and deployment using a REST API.

⸻

🎯 Objective

To build a time-series forecasting system that predicts the next day’s adjusted closing price based on historical stock data and engineered features.

⸻

🧠 Model
	•	Model: LSTM (Long Short-Term Memory)
	•	Framework: TensorFlow / Keras
	•	Input Shape: (20 timesteps, 14 features)
	•	Output: Next day Adjusted Close price

⸻

📊 Features Used

The model is trained using engineered features derived from raw stock data:
	•	Open
	•	Volume
	•	HL Range (High - Low)
	•	Volume-5 (5-day rolling average)
	•	Month
	•	Month_sin, Month_cos (cyclical encoding)
	•	Day of week
	•	dow_sin (cyclical encoding)
	•	Returns
	•	Volatility_5, Volatility_10
	•	SMA-10 (Simple Moving Average)
	•	Adj Close (target feature used during training)

⸻

⚙️ Feature Engineering

The following transformations are applied:
	•	Rolling statistics (SMA, volatility)
	•	Percentage returns
	•	Time-based features (month, day of week)
	•	Cyclical encoding using sine and cosine
	•	Data normalization using MinMaxScaler

⸻

🏗️ Project Structure
.
├── best_lstm_model.h5
├── dataset_scaler.pkl
├── target_scaler.pkl
├── features.pkl
├── server.py
├── requirements.txt
└── README.md

⸻

🌐 API Deployment

The model is deployed using FastAPI.

🔹 Endpoint
POST /predict
🔹 Input
	•	Upload a CSV file containing stock data with columns:
  Date, Open, High, Low, Adj Close, Volume
  	•	Minimum rows required: ≥ 20
🔹 Output
{
  "predicted_adj_close": 245.67
}

⸻

🔄 Prediction Workflow

CSV Input
   ↓
Feature Engineering
   ↓
Handle Missing Values
   ↓
Select Last 20 Timesteps
   ↓
Scaling
   ↓
LSTM Model
   ↓
Prediction (Next Day Price)

⸻

📈 Evaluation Metrics
	•	RMSE (Root Mean Squared Error)
	•	MAE (Mean Absolute Error)
	•	MAPE (Mean Absolute Percentage Error)

