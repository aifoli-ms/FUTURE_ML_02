import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Download dataset from Kaggle

df = pd.read_csv(f"/workspaces/FUTURE_ML_02/AAPL.csv")

# Display basic information
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Feature Engineering
def create_features(data, target_col='Close', window=30):
    df_features = data.copy()
    
    # Technical indicators
    df_features['Returns'] = df_features[target_col].pct_change()
    df_features['MA7'] = df_features[target_col].rolling(window=7).mean()
    df_features['MA30'] = df_features[target_col].rolling(window=30).mean()
    df_features['RSI'] = calculate_rsi(df_features[target_col])
    df_features['Volatility'] = df_features['Returns'].rolling(window=window).std()
    
    # Target variable (next day's closing price)
    df_features['Target'] = df_features[target_col].shift(-1)
    
    return df_features.dropna()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Create features
df_processed = create_features(df)

# Prepare data for LSTM
feature_columns = ['Close', 'Returns', 'MA7', 'MA30', 'RSI', 'Volatility']
target_column = 'Target'

# Scale the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_processed[feature_columns])
scaled_target = scaler.fit_transform(df_processed[[target_column]])

# Create sequences for LSTM
# This function creates sequences of historical data points for LSTM training
def create_sequences(data, target, seq_length=60):
    X, y = [], []
    # Loop through the data, creating sequences of length 'seq_length'
    # Each sequence will be used to predict the next target value
    for i in range(len(data) - seq_length):
        # Create sequence of 'seq_length' time steps (e.g. 60 days of features)
        X.append(data[i:(i + seq_length)])
        # The target is the next value after the sequence
        y.append(target[i + seq_length])
    # Convert lists to numpy arrays for model training
    return np.array(X), np.array(y)

# Use 60 days of historical data to predict the next day
seq_length = 60
# Create sequences from our scaled feature and target data
# X will have shape (num_sequences, seq_length, num_features)
# y will have shape (num_sequences, 1) containing the target values
X, y = create_sequences(scaled_features, scaled_target, seq_length)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
# Build a sequential LSTM model for time series prediction
model = Sequential([
    # First LSTM layer with 100 units
    # return_sequences=True allows output to be fed into next LSTM layer
    # input_shape specifies the expected input dimensions: (sequence length, number of features)
    LSTM(100, return_sequences=True, input_shape=(seq_length, len(feature_columns))),
    
    # Dropout layer to prevent overfitting by randomly dropping 20% of units
    Dropout(0.2),
    
    # Second LSTM layer with 50 units
    # return_sequences=False since this is the last LSTM layer
    LSTM(50, return_sequences=False),
    
    # Another dropout layer
    Dropout(0.2),
    
    # Final dense layer with 1 output unit for price prediction
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions for actual prices
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Calculate metrics
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print("\nModel Performance Metrics:")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Root Mean Square Error: ${rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Plot actual vs predicted prices
plt.figure(figsize=(15, 6))
plt.plot(df_processed['Date'].iloc[train_size+seq_length:].reset_index(drop=True), y_test_inv, label='Actual Price')
plt.plot(df_processed['Date'].iloc[train_size+seq_length:].reset_index(drop=True), y_pred_inv, label='Predicted Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stock_prediction.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test_inv[:100], label='Actual')
plt.plot(y_pred_inv[:100], label='Predicted')
plt.title('First 100 Predictions vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('stock_training_history.png')
plt.close()

# Save predictions to CSV
predictions_df = pd.DataFrame({
    'Date': df_processed['Date'].iloc[train_size+seq_length:].reset_index(drop=True),
    'Actual_Price': y_test_inv.flatten(),
    'Predicted_Price': y_pred_inv.flatten(),
    'Absolute_Error': np.abs(y_test_inv.flatten() - y_pred_inv.flatten())
})
predictions_df.to_csv('stock_predictions.csv', index=False)

print("\nResults have been saved to:")
print("- stock_prediction.png")
print("- stock_training_history.png")
print("- stock_predictions.csv")