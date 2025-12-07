import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Download gold futures data
ticker = "GC=F"
start_date = "2018-01-01"
end_date = None

df = yf.download(ticker, start=start_date, end=end_date)
df.head()

# Function to create simple lag features
def create_simple_features(df):
    data = pd.DataFrame()
    
    close_prices = df["Close"]

    data["TodayPrice"] = close_prices
    data["PreviousDayPrice"] = close_prices.shift(1)
    data["TwoDaysAgoPrice"] = close_prices.shift(2)
    data["ThreeDaysAgoPrice"] = close_prices.shift(3)

    data = data.dropna()

    feature_cols = ["PreviousDayPrice", "TwoDaysAgoPrice", "ThreeDaysAgoPrice"]

    X = data[feature_cols].values
    y = data["TodayPrice"].values

    return X, y, data, feature_cols


# Create features
X, y, data, feature_cols = create_simple_features(df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model training complete with updated data!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Model Performance on Test Set:")
print("RMSE:", rmse)
print("MAE :", mae)

# Plot actual vs predicted prices
test_dates = data.index[-len(y_test):]

plt.figure(figsize=(12,6))
plt.plot(test_dates, y_test, label="Actual Gold Price")
plt.plot(test_dates, y_pred, label="Predicted Gold Price", linestyle="--")
plt.title("Actual vs Predicted Gold Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# Predict the next-day gold price
last_row = data.iloc[-1]
last_features = last_row[["PreviousDayPrice", "TwoDaysAgoPrice", "ThreeDaysAgoPrice"]].values.reshape(1, -1)

next_day_prediction = model.predict(last_features)[0]
print(f"Predicted next-day closing price for {ticker}: {next_day_prediction:.2f} USD")
