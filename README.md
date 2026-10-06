# energy-price-predictions
Predicting the day ahead price of energy using techniques from linear algebra and maschine learning

## 1) Get the data
In the data directory, running the notebook *data.ipynb* pulls real power data. Note that you need an ENTSOE API key for this. Alternatively, run  *create_dummy_csv.py* to create a .csv file with fake data. Both ways format the data accordingly, for example converting hours into a periodic format that can be exploited by deep learning algorithms.

## 2) Analyze the data
Once the .csv exists, running *price_predictions.jl* does a ridge regression, shows the coefficients and makes a dashboard of useful plots.

![Ridge Regression Plots](ridge_regression_plots_lambda_5.png)
