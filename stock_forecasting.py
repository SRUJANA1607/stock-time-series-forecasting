# ===============================
# Stock Market Time-Series Forecasting
# Models: AR, MA, ARIMA, SARIMA, SARIMAX
# ===============================

# -------- Basic imports --------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# -------- Create synthetic dataset (ONLY if CSV not present) --------
np.random.seed(42)

dates = pd.date_range(start="2018-01-01", end="2024-01-01", freq="B")
n = len(dates)

price = 100 + np.cumsum(np.random.normal(0, 0.3, n))
volume = np.random.randint(9e5, 13e5, n)

df = pd.DataFrame({
    "date": dates,
    "close": price,
    "volume": volume
})

df.to_csv("stock_data1.csv", index=False)

# -------- Load dataset --------
df = pd.read_csv("stock_data1.csv", parse_dates=["date"], index_col="date")
df = df.asfreq("B")
df["close"].interpolate(inplace=True)

# -------- Quick EDA --------
print(df.shape)
print(df[["close", "volume"]].describe())

plt.figure(figsize=(12,4))
plt.plot(df["close"])
plt.title("Synthetic Close Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.show()

# -------- Train-test split --------
train_pct = 0.8
split = int(len(df) * train_pct)

train = df.iloc[:split]
test = df.iloc[split:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# -------- Evaluation function --------
def evaluate_series(true, pred, label="Model"):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    print(f"{label} --> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# ===============================
# 1) Autoregressive (AR)
# ===============================
p = 5
ar_model = AutoReg(train["close"], lags=p, old_names=False).fit()
ar_forecast = ar_model.predict(start=test.index[0], end=test.index[-1])

evaluate_series(test["close"], ar_forecast, "AR(p=5)")

plt.figure(figsize=(12,4))
plt.plot(train["close"].iloc[-200:], label="Train (last 200)")
plt.plot(test["close"], label="Test")
plt.plot(ar_forecast, label="AR Forecast")
plt.legend()
plt.show()

# ===============================
# 2) Moving Average (MA) via ARIMA(0,0,q)
# ===============================
q = 10
ma_model = ARIMA(train["close"], order=(0,0,q)).fit()
ma_forecast = ma_model.predict(start=test.index[0], end=test.index[-1])

evaluate_series(test["close"], ma_forecast, f"MA(q={q})")

plt.figure(figsize=(12,4))
plt.plot(test["close"], label="Test")
plt.plot(ma_forecast, label="MA Forecast")
plt.legend()
plt.show()

# ===============================
# 3) ARIMA (p,d,q)
# ===============================
p, d, q = 5, 1, 2
arima_model = ARIMA(train["close"], order=(p,d,q)).fit()
arima_forecast = arima_model.predict(start=test.index[0], end=test.index[-1], typ="levels")

evaluate_series(test["close"], arima_forecast, "ARIMA(5,1,2)")

plt.figure(figsize=(12,4))
plt.plot(test["close"], label="Test")
plt.plot(arima_forecast, label="ARIMA Forecast")
plt.legend()
plt.show()

# ===============================
# 4) SARIMA
# ===============================
sarima_model = SARIMAX(
    train["close"],
    order=(1,1,1),
    seasonal_order=(1,1,1,5)
).fit(disp=False)

sarima_forecast = sarima_model.predict(
    start=test.index[0],
    end=test.index[-1],
    typ="levels"
)

evaluate_series(test["close"], sarima_forecast, "SARIMA")

plt.figure(figsize=(12,4))
plt.plot(test["close"], label="Test")
plt.plot(sarima_forecast, label="SARIMA Forecast")
plt.legend()
plt.show()

# ===============================
# 5) SARIMAX with exogenous variable (volume)
# ===============================
train_exog = np.log1p(train["volume"])
test_exog = np.log1p(test["volume"])

sarimax_model = SARIMAX(
    train["close"],
    exog=train_exog,
    order=(1,1,1),
    seasonal_order=(1,0,1,5)
).fit(disp=False)

sarimax_forecast = sarimax_model.predict(
    start=test.index[0],
    end=test.index[-1],
    exog=test_exog,
    typ="levels"
)

evaluate_series(test["close"], sarimax_forecast, "SARIMAX with volume")

plt.figure(figsize=(12,4))
plt.plot(test["close"], label="Test")
plt.plot(sarimax_forecast, label="SARIMAX Forecast")
plt.legend()
plt.show()

# -------- Save forecasts --------
out = pd.DataFrame({
    "true": test["close"],
    "ar_forecast": ar_forecast,
    "ma_forecast": ma_forecast,
    "arima_forecast": arima_forecast,
    "sarima_forecast": sarima_forecast,
    "sarimax_forecast": sarimax_forecast
})

out.to_csv("model_forecasts.csv")
print("Saved model_forecasts.csv to current working directory.")
