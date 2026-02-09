import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Stock Forecasting", layout="wide")

st.title("📈 Stock Market Time-Series Forecasting")
st.write("Models: AR, MA, ARIMA, SARIMA, SARIMAX")

# Generate synthetic data
np.random.seed(42)
dates = pd.date_range("2018-01-01", "2024-01-01", freq="B")
price = 100 + np.cumsum(np.random.normal(0, 0.3, len(dates)))
volume = np.random.randint(9e5, 13e5, len(dates))

df = pd.DataFrame({"close": price, "volume": volume}, index=dates)

st.subheader("Synthetic Stock Price")
st.line_chart(df["close"])

# Train-test split
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

def evaluate(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    return rmse, mae

model_choice = st.selectbox(
    "Choose Forecasting Model",
    ["AR", "MA", "ARIMA", "SARIMA", "SARIMAX"]
)

if model_choice == "AR":
    model = AutoReg(train["close"], lags=5).fit()
    forecast = model.predict(start=test.index[0], end=test.index[-1])

elif model_choice == "MA":
    model = ARIMA(train["close"], order=(0,0,10)).fit()
    forecast = model.predict(start=test.index[0], end=test.index[-1])

elif model_choice == "ARIMA":
    model = ARIMA(train["close"], order=(5,1,2)).fit()
    forecast = model.predict(start=test.index[0], end=test.index[-1], typ="levels")

elif model_choice == "SARIMA":
    model = SARIMAX(
        train["close"],
        order=(1,1,1),
        seasonal_order=(1,1,1,5)
    ).fit(disp=False)
    forecast = model.predict(start=test.index[0], end=test.index[-1], typ="levels")

else:
    train_exog = np.log1p(train["volume"])
    test_exog = np.log1p(test["volume"])
    model = SARIMAX(
        train["close"],
        exog=train_exog,
        order=(1,1,1),
        seasonal_order=(1,0,1,5)
    ).fit(disp=False)
    forecast = model.predict(
        start=test.index[0],
        end=test.index[-1],
        exog=test_exog,
        typ="levels"
    )

rmse, mae = evaluate(test["close"], forecast)

st.subheader(f"{model_choice} Forecast")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(test["close"], label="Actual")
ax.plot(forecast, label="Forecast")
ax.legend()
st.pyplot(fig)

st.success(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")
