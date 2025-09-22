from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================================
# 1. Prepare Data
# ================================
values = data[['Close']].values   # only close price
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values)

training_data_len = int(len(values) * 0.8)
train_data = scaled_data[:training_data_len]
test_data  = scaled_data[training_data_len-60:]

# Training sequences
X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Test sequences
X_test, y_test = [], values[training_data_len:]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# ================================
# 2. Build Model
# ================================
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# ================================
# 3. Train
# ================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ================================
# 4. Save trained weights
# ================================
model.save_weights("lstmN.weights.h5")
print("✅ Weights saved as lstmN.weights.h5")

# ================================
# 5. Make Predictions
# ================================
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

data = data.reset_index()   # move Date into a column
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# Train/test split
train = data[:training_data_len]
test = data[training_data_len:].copy()
test['Predictions'] = predictions

# Plot results
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Close'], label="All Data (Train + Test)", color="green")
plt.plot(test.index, test['Close'], label="Actual Test", color="blue")
plt.plot(test.index, test['Predictions'], label="Predicted", color="red")
plt.title("LSTM Stock Price Prediction (Close only)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
