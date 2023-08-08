import pandas as pd
pd.options.plotting.backend = "plotly"

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pmdarima as pm
from pmdarima.arima import auto_arima
from pmdarima.arima import ndiffs

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import itertools
from tqdm import tqdm

import mlflow

def Setting():
    # 전역으로 그래프 사이즈 고정
    plt.rcParams["figure.figsize"] = (12,5)

    # 유니코드 깨짐현상 해결
    plt.rcParams['axes.unicode_minus'] = False
    
    # 나눔고딕 폰트 적용
    plt.rcParams["font.family"] = 'NanumGothic'
Setting()

# Step 1: Load the data
debt_cycle_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/데이터자료/debt_cycle.csv')

# Step 2: Preprocess the data (if needed)
debt_cycle_df.loc[:,'Date'] = pd.to_datetime(debt_cycle_df.Date)
debt_cycle_df = debt_cycle_df.set_index('Date')
debt_cycle_df = debt_cycle_df.drop(['Debt', 'y'], axis = 1)

# split data
forecast_year = 20 # 마지막 forecast_year 년 예측
df_train = debt_cycle_df.iloc[:-forecast_year]
df_test = debt_cycle_df.iloc[len(df_train):]

# For simplicity, we assume the data is already preprocessed and stationary.
# Set the active experiment
mlflow.set_experiment("SARIMAX_m75_forecast")
mlflow.autolog()
# Step 3: Define the ARIMA model
# Define a list of combinations for p, d, and q that you want to try
p_values = [0, 1, 2]
d_values = [1, 2]
q_values = [0, 1, 2]

# Loop through different combinations of p, d, and q
for p in p_values:
    for d in d_values:
        for q in q_values:
            # Skip combinations where d=0 (non-stationary)
            if d == 0:
                continue
            
            # Define the SARIMAX model
            order = (p, d, q)  # ARIMA order (p, d, q)
            seasonal_order = (1, 1, 1, 75)  # Seasonal order (P, D, Q, S)
            
            # If you have exogenous variables, include them in the 'exog' parameter
            exog_train_data = None   # Replace with your exogenous training data
            exog_test_data = None    # Replace with your exogenous test data
            
            # Start a new MLflow run
            with mlflow.start_run() as run :
                # Define the SARIMAX model
                model_sarimax = sm.tsa.SARIMAX(debt_cycle_df, order=order, seasonal_order = seasonal_order, exog = exog_train_data)
                results_sarimax = model_sarimax.fit()
                
                # Make predictions using the SARIMAX model
                predictions_sarimax = results_sarimax.predict(start=len(df_train), end=len(debt_cycle_df), exog = exog_test_data)
                forecast_sarimax = results_sarimax.forecast(step = forecast_year)
                
                # Evaluate the model using MAPE
                mape_sarimax = np.mean(np.abs((df_test.Cycle - predictions_sarimax) / df_test.Cycle)) * 100
                
                # Log SARIMAX hyperparameters and metrics for this combination
                mlflow.log_param("SARIMAX_p", p)
                mlflow.log_param("SARIMAX_d", d)
                mlflow.log_param("SARIMAX_q", q)
                mlflow.log_param("Seasonal_P", seasonal_order[0])
                mlflow.log_param("Seasonal_D", seasonal_order[1])
                mlflow.log_param("Seasonal_Q", seasonal_order[2])
                mlflow.log_param("Seasonal_S", seasonal_order[3])
                # mlflow.log_metric("MAPE", mape_sarimax)
                
                # Save the model using MLflow's model logging capability
                plot_filename = "sarimax_forecasting_plot.png"
                plt.plot(debt_cycle_df, label = "Actual", color = 'r')
                plt.plot(forecast_sarimax, label = "Forecasted", color = 'b')
                plt.xlabel("Date")
                plt.ylabel("%")
                plt.title("SARIMAX FORECAST")
                plt.legend()
                plt.grid()
                plt.savefig(plot_filename)
                plt.clf()
                # Log the plot as an artifact in MLflow
                mlflow.log_artifact(plot_filename)
# End the MLflow run
mlflow.end_run()
