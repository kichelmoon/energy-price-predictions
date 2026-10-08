# energy-price-predictions
Predicting the day ahead price of energy using techniques from linear algebra and maschine learning

## 1) Get the data
In the data directory, running the notebook *data.ipynb* pulls real power data. Note that you need an ENTSOE API key for this. Alternatively, run  *create_dummy_csv.py* to create a .csv file with fake data. Both ways format the data accordingly, for example converting hours into a periodic format that can be exploited by deep learning algorithms.

## 2) Analyze the data
Once the .csv exists, there are two approaches to analyze the data and make predictions: A classical ridge regression and a simple neural network with RELU and Stochastic Gradient Descent. The file *energy_price_predictions.jl* runs both methods and builds dashboards that can be used to view some of the model's properties.
![Ridge Regression Plots](plots/ridge_regression_plots_lambda_5.png)
![Deep Learning Plots](plots/deep_learning_plots_epochs_100.png)

# ⚡ Day Ahead Electricity Price Forecasting ⚡

Quantitative time-series analysis and machine learning pipeline to forecast EPEX SPOT electricity prices using fundamental market drivers (renewable generation, load forecasts, and historical price data).

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Project Overview

In liberalized electricity markets, accurate price forecasting is essential for trading desks, portfolio managers, and renewable energy aggregators. This project implements and compares multiple quantitative models to predict **[z. B. Day-Ahead / Intraday]** electricity prices for the **[z. B. German/Austrian (DE/LU)]** bidding zone.

### Key Objectives
* Fetch and preprocess market data from public sources (ENTSO-E Transparency Platform / SMARD).
* Perform feature engineering capturing temporal dynamics (seasonality, hour-of-day, day-of-week) and fundamental drivers.
* Train and evaluate time-series and machine learning forecasting models.
* Benchmarking model performance using domain-standard evaluation metrics.

---

## 📊 Data & Pipeline Architecture

### Data Sources
* **Prices:** Historical Day-Ahead Auction prices (€/MWh).
* **Generation & Load:** Actual & forecasted wind (onshore/offshore), solar generation, and total load demand.

### Workflow
1. **Data Ingestion & Cleaning:** Automated fetching, missing value imputation, and timezone alignment (UTC to CET/CEST).
2. **Feature Engineering:** Lagged features (t-24, t-168), rolling statistics, price spreads, and calendar indicators.
3. **Modeling:**
   * **Baseline:** Naïve Persistence Model (t-24 / t-168)
   * **Statistical:** ARIMA / SARIMAX
   * **Machine Learning:** LightGBM / XGBoost / Random Forest
4. **Evaluation:** Out-of-sample backtesting on a rolling/expanding window.

---

## 📈 Key Results & Performance

Evaluation metric comparisons on the test set:

| Model | MAE (€/MWh) | RMSE (€/MWh) | WAPE (%) |
| :--- | :---: | :---: | :---: |
| Naïve Benchmark | 14.20 | 19.50 | 15.3% |
| SARIMAX | 9.80 | 13.10 | 10.5% |
| **LightGBM (Best)** | **6.40** | **9.10** | **6.8%** |

*(Ersetze die Zahlen oben durch deine echten Modellergebnisse)*

> **Visual Results:**
> *(Hier einen Screenshot/Plot deines Modells einfügen: z. B. `![Model Predictions](docs/price_forecast_plot.png)`)*

---

## 🛠 Tech Stack & Libraries

* **Language:** Python
* **Data Processing:** `pandas`, `numpy`
* **Modeling & ML:** `scikit-learn`, `lightgbm`, `xgboost`, `statsmodels`
* **Visualization:** `matplotlib`, `seaborn`
* **Version Control:** Git

---

## 🚀 Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/kichelmoon/energy-price-predictions.git](https://github.com/kichelmoon/energy-price-predictions.git)
   cd energy-price-predictions