# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import shap
import os

# -------------------------------
# 1. Build Model Architecture
# -------------------------------

def build_final_model(lookback=60, n_features=1):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
        LSTM(64, return_sequences=False),
        Dense(128, activation="relu"),
        Dense(1)   # final output for Close price
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


@st.cache_resource
def load_trained_model(weights_path="lstmN.weights.h5"):
    model = build_final_model()
    model.load_weights(weights_path)
    return model

model = load_trained_model()

# -------------------------------
# 2. Helper Functions
# -------------------------------
def create_sequences(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def forecast_next_days(model, scaler, df, lookback=60, horizon=5):
    last_window = df['Close'].values[-lookback:].reshape(-1,1)
    last_scaled = scaler.transform(last_window)

    predictions, dates = [], []
    current_input = last_scaled.copy()

    for i in range(horizon):
        X = current_input.reshape(1, lookback, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0,0]
        predictions.append(pred_price)
        dates.append(df.index[-1] + pd.tseries.offsets.BDay(i+1))

        # update input
        current_input = np.append(current_input[1:], pred_scaled).reshape(-1,1)

    return pd.DataFrame({"Date": dates, "Predicted Close": predictions})

# -------------------------------
# 3. Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Stock Price Prediction (LSTM)", layout="wide")
st.title("📈 LSTM Stock Price Prediction (Close Only)")

# Sidebar
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2015,1,1))
end_date = st.sidebar.date_input("End Date", datetime.today())
lookback = st.sidebar.slider("Lookback (days)", 30, 120, 60)
horizon = 5

# Load Data
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    st.error("No data found for this ticker/date range.")
    st.stop()

st.subheader(f"Raw Data for {ticker}")
st.write(data.tail())

# Scale
values = data[['Close']].values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values)

# Sequences
X, y = create_sequences(scaled_data, lookback)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Load Model
model = load_trained_model()

# -------------------------------
# 4. Prediction & Backtest
# -------------------------------
if st.sidebar.button("Run Prediction"):
    st.subheader("Backtest on Test Data")

    preds = model.predict(X_test)
    preds_rescaled = scaler.inverse_transform(preds)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

    test_dates = data.index[-len(y_test):]
    test_df = pd.DataFrame({"Date": test_dates, "Actual": y_test_rescaled.flatten(),
                            "Predicted": preds_rescaled.flatten()})
    test_df = test_df.set_index("Date")

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data.index, data['Close'], label="Historical Close", color="green")
    ax.plot(test_df.index, test_df["Actual"], label="Actual Test", color="blue")
    ax.plot(test_df.index, test_df["Predicted"], label="Predicted", color="red")
    ax.set_title(f"{ticker} Close Price Prediction")
    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # 5. Forecast Future
    # -------------------------------
    st.subheader(f"{horizon}-Day Forecast")
    forecast_df = forecast_next_days(model, scaler, data, lookback, horizon)
    st.write(forecast_df)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data.index, data['Close'], label="History")
    ax.plot(forecast_df["Date"], forecast_df["Predicted Close"],
            marker="o", label="Forecast")
    ax.set_title(f"Next {horizon}-Day Forecast for {ticker}")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # 6. Explainable AI
    # -------------------------------
st.subheader("Explainable AI (SHAP Feature Importance)")
try:
    # Build last sequence (same input shape as model expects)
    last_window = scaled_data[-lookback:]
    seq = np.expand_dims(last_window, axis=0)  # (1, lookback, 1)

    # Use a background sample for SHAP
    background = X_train[np.random.choice(len(X_train), size=50, replace=False)]

    explainer = shap.Explainer(model, background)
    shap_values = explainer(seq)

    # Plot feature importance
    fig = plt.figure()
    shap.summary_plot(shap_values, seq, plot_type="bar", show=False)
    st.pyplot(fig)

except Exception as e:
    st.warning("⚠️ SHAP explanation not available in this environment.")
    st.text(str(e))

    # -------------------------------
    # 7. Narrative
    # -------------------------------
    st.subheader("AI Narrative")
    st.info(
        "This LSTM model predicts only the **Close Price**. "
        "A rising forecast compared to the historical trend suggests potential bullish momentum. "
        "Use forecasts with caution as they depend heavily on recent price patterns."
    )



