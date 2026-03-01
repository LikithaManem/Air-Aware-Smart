# =====================================
# AIRAWARE SMART - Milestone 2
# AQI Prediction using Prophet
# =====================================

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# -----------------------------
# 1. Load AQI Data
# -----------------------------

df = pd.read_csv(r"c:\Users\hp\OneDrive\Desktop\aqi_data.csv.txt")

print("Original Data:")
print(df.head())


# -----------------------------
# 2. Preprocessing
# -----------------------------

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Fill missing values (if any)
df["AQI"] = df["AQI"].fillna(df["AQI"].mean())

# Sort data by date
df = df.sort_values("Date")

# Rename columns for Prophet
df = df.rename(columns={"Date": "ds", "AQI": "y"})

print("\nAfter Preprocessing:")
print(df.head())


# -----------------------------
# 3. Basic Analysis
# -----------------------------

print("\nSummary Statistics:")
print(df["y"].describe())


# -----------------------------
# 4. Plot Original Data
# -----------------------------

plt.plot(df["ds"], df["y"])
plt.title("Original AQI Trend")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.show()


# -----------------------------
# 5. Train Model
# -----------------------------

model = Prophet()
model.fit(df)


# -----------------------------
# 6. Create Future Dates
# -----------------------------

future = model.make_future_dataframe(periods=7)

print("\nFuture Dates:")
print(future.tail())


# -----------------------------
# 7. Make Predictions
# -----------------------------

forecast = model.predict(future)

print("\nPredicted AQI:")
print(forecast[["ds", "yhat"]].tail(7))


# -----------------------------
# 8. Save Predictions
# -----------------------------

forecast[["ds", "yhat"]].to_csv("predicted_aqi.csv", index=False)


# -----------------------------
# 9. Plot Forecast
# -----------------------------

model.plot(forecast)
plt.title("AQI Prediction")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.show()


# -----------------------------
# 10. Show Components
# -----------------------------

model.plot_components(forecast)
plt.show()