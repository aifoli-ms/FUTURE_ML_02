# FUTURE_ML_02

# 📈 AAPL Stock Price Prediction with LSTM

This project uses an LSTM (Long Short-Term Memory) deep learning model to predict the next day's **Apple Inc. (AAPL)** stock closing price based on historical data and technical indicators.

## 🗂️ Project Structure

- `AAPL.csv`: Historical stock data downloaded from Kaggle.
- `main.py`: Python script that:
  - Loads and preprocesses data
  - Creates technical features like Moving Averages, RSI, and Volatility
  - Scales the features using MinMaxScaler
  - Builds and trains an LSTM model
  - Evaluates model performance
  - Plots results and saves predictions
- `stock_prediction.png`: Line plot comparing actual vs. predicted prices.
- `stock_training_history.png`: Training and validation loss curves + sample predictions.
- `stock_predictions.csv`: CSV with actual price, predicted price, and absolute error for the test set.

## 📊 Features Used

- **Close Price**
- **Returns** (Percentage change)
- **MA7 & MA30** (Moving averages)
- **RSI** (Relative Strength Index)
- **Volatility** (Rolling standard deviation)

## 🧠 Model Overview

- A two-layer LSTM neural network with dropout to prevent overfitting.
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Input Sequence: Past 60 days of stock indicators
- Output: Next day's closing price

## ⚙️ How to Run

1. Clone this repository or upload the script and dataset to your environment.
2. Ensure the required libraries are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow