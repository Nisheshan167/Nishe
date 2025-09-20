import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# -------------------------------
# Helper functions
# -------------------------------
def create_sequences(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def predict_next_five_days(model, scaler, df, lookback=60):
    last_window = df['Close'].values[-lookback:].reshape(-1, 1)
    last_window_scaled = scaler.transform(last_window)
    forecast_prices, forecast_dates = [], []
    current_input = last_window_scaled.copy()

    for i in range(5):  # Forecast 5 steps
        X = current_input.reshape(1, lookback, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
        forecast_prices.append(pred_price)
        forecast_dates.append(df.index[-1] + pd.tseries.offsets.BDay(i+1))
        current_input = np.append(current_input[1:], pred_scaled, axis=0)

    return forecast_dates, forecast_prices

# -------------------------------
# Sidebar inputs
# -------------------------------
st.sidebar.title("📈 LSTM Stock Predictor")

ticker = st.sidebar.text_input("Ticker", "MSFT")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(datetime.today().strftime("%Y-%m-%d")))
lookback = st.sidebar.slider("Lookback window (days)", 30, 120, 60)

# -------------------------------
# Download data
# -------------------------------
st.write(f"### Data for {ticker}")
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error("No data found for this ticker/date range.")
    st.stop()

df['Date'] = pd.to_datetime(df.index)
st.write(df.tail())

# Plot closing price
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['Close'], label="Close Price")
ax.set_title(f"{ticker} Closing Price")
ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
st.pyplot(fig)

# -------------------------------
# Train Model
# -------------------------------
if st.sidebar.button("Train Model"):
    st.write("### Training LSTM Model...")

    # Scale data
    close_data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_data)

    # Sequences
    X, y = create_sequences(scaled_data, lookback)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions))
    mae = mean_absolute_error(y_test_rescaled, predictions)
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")

    # Plot actual vs predicted
    st.write("### Backtest Results")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index[-len(y_test):], y_test_rescaled, label="Actual")
    ax.plot(df.index[-len(y_test):], predictions, label="Predicted")
    ax.set_title("Actual vs Predicted Close Price")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Forecast
    # -------------------------------
    st.write("### 5-Day Forecast")
    future_dates, future_preds = predict_next_five_days(model, scaler, df, lookback)

    forecast_df = pd.DataFrame({
        "Date": [d.date() for d in future_dates],
        "Predicted Close": [round(p,2) for p in future_preds]
    })
    st.write(forecast_df)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df['Close'], label="History")
    ax.plot(future_dates, future_preds, marker="o", label="Forecast")
    ax.set_title(f"Next 5-Day Forecast for {ticker}")
    ax.legend()
    st.pyplot(fig)
