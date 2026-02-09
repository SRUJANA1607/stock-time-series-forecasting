Stock Market Time-Series Forecasting System

Project Overview
This project implements a Stock Market Time-Series Forecasting System using classical statistical time-series models. The objective is to analyze historical stock price patterns and forecast future values by comparing multiple forecasting algorithms.
The project was developed as part of a Data Analytics course and deployed as an interactive web application using Streamlit.

Problem Statement
Stock market prices change over time and exhibit trends, seasonality, and randomness. Accurate forecasting helps in understanding market behavior and supports informed decision-making. This project focuses on forecasting stock prices using time-series models and evaluating their performance.

Existing System
Traditional stock market forecasting systems typically rely on single statistical models such as Autoregressive or ARIMA-based approaches.

Limitations of the Existing System
Existing systems often struggle with capturing seasonality, require manual parameter tuning, and provide limited comparative analysis across different models. Their performance may degrade in highly volatile market conditions.

Proposed System
The proposed system implements multiple time-series forecasting models on the same dataset and performs a comparative analysis to identify the most suitable model for stock price prediction.

Models Implemented
Autoregressive (AR) model
Moving Average (MA) model
ARIMA model
SARIMA model
SARIMAX model with exogenous variable (trading volume)

Key Features
Comparison of multiple time-series forecasting models
Evaluation using RMSE and MAE metrics
Visualization of actual versus predicted stock prices
Interactive model selection through a web interface
End-to-end workflow from data generation to forecasting and evaluation

Dataset
The project uses synthetic stock market time-series data consisting of closing price and trading volume. Synthetic data was chosen to ensure controlled experimentation and consistent comparison across different forecasting models without dependency on external data sources.

Tech Stack
Programming Language: Python
Libraries: pandas, numpy, matplotlib, statsmodels, scikit-learn, streamlit
Development Environment: VS Code
Deployment Platform: Streamlit Community Cloud

Live Application
Deployed application link:
https://stock-time-series-forecasting-ywjbxqxtwzlh9aegmq93c4.streamlit.app/

How to Run Locally
Install required dependencies using requirements.txt.
Run the application using Streamlit.

Evaluation Metrics
Root Mean Square Error (RMSE)
Mean Absolute Error (MAE)

Results and Observations
The project demonstrates that seasonal models such as SARIMA and SARIMAX perform better when seasonality is present in the data. The SARIMAX model benefits from including an external variable such as trading volume. Comparative analysis provides better insight than using a single forecasting model.

Conclusion
This project highlights the importance of time-series analysis in stock market forecasting. By implementing and comparing multiple forecasting models, the system effectively identifies the strengths and limitations of each approach and provides meaningful analytical insights.

References
Time Series Analysis using Statsmodels
ARIMA and SARIMA forecasting models
Online learning resources and tutorials on time-series forecasting
