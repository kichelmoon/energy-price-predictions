import numpy as np
import pandas as pd

number_of_samples = 1000

date_range = pd.date_range(
    start="2024-01-01 00:00", periods=number_of_samples, freq="1h", tz="Europe/Berlin"
)

hours = date_range.hour

solar_pattern = np.maximum(0, np.sin((hours - 7) * np.pi / 9))
solar_forecast = np.clip(
    solar_pattern * 8000 + np.random.normal(0, 300, number_of_samples), 0, None
)

wind_base = 25000 + 10000 * np.sin(np.linspace(0, 4 * np.pi, number_of_samples))
wind_forecast = np.clip(
    wind_base + np.random.normal(0, 1500, number_of_samples), 5000, 45000
)

load_pattern = (
    50000
    + 15000 * np.sin((hours - 6) * np.pi / 12) ** 2
    + np.random.normal(0, 1000, number_of_samples)
)
total_load = np.clip(load_pattern, 40000, 75000)

base_price = 75.0
price_variance = (
    (total_load - 55000) / 1000
    - (solar_forecast + wind_forecast) / 1500
    + np.random.normal(0, 8, number_of_samples)
)
energy_prices = (base_price + price_variance).round(2)

df_hourly = pd.DataFrame(
    {
        "energy_prices": energy_prices,
        "total_load": total_load.round(1),
        "day_ahead_solar_forecast": solar_forecast.round(1),
        "wind_forecast": wind_forecast.round(1),
    },
    index=date_range,
)

# Zyklische Daten für Stunden und Position im Jahr
hours = df_hourly.index.hour
df_hourly["hour_sin"] = np.sin(2 * np.pi * hours / 24)
df_hourly["hour_cos"] = np.cos(2 * np.pi * hours / 24)

days_in_year = np.where(df_hourly.index.is_leap_year, 366, 365)
day_of_year = df_hourly.index.dayofyear
df_hourly["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
df_hourly["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)


#One Hot Encoding für Wochentag und Monat
weekday_dummies = pd.get_dummies(
    df_hourly.index.dayofweek, prefix="weekday", dtype=int
)
weekday_dummies.index = df_hourly.index  # Index ausrichten

month_dummies = pd.get_dummies(
    df_hourly.index.month, prefix="month", dtype=int
)
month_dummies.index = df_hourly.index  # Index ausrichten


df_prepared = pd.concat([df_hourly, weekday_dummies, month_dummies], axis=1)

print("--- Form des transformierten DataFrames ---")
print(f"Zeilen: {df_prepared.shape[0]}, Spalten: {df_prepared.shape[1]}\n")

print("--- Neue Spaltenübersichten ---")
print(df_prepared.columns.tolist())

print("\n--- Erste 5 Zeilen ---")
print(df_prepared.head())
df_prepared.to_csv('entsoe_hourly_data_de_lu.csv')