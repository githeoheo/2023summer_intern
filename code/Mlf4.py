### arima코드에 사이클 외에 다른 데이터 넣어보기
### ARIMA에 기존 사이클 데이터를 사용하였을 때 값이 원하는대로 나오지 않아서 나스닥 주가 데이터를 사용해 보기로 했다.
### 결과는 기존 사이클 데이터와 비슷하게 나왔고 ARIMA 결과 원인을 알 수 있게 되었다.

import pandas as pd
# pd.options.plotting.backend = "plotly"

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
    plt.rcParams["font.family"] = 'Malgun Gothic'
Setting()

# Step 1: Load the data
debt_cycle_df = pd.read_csv('C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/나스닥(1985~2023)_yfinance.csv')


# Step 2: Preprocess the data (if needed)
debt_cycle_df.loc[:,'Date'] = pd.to_datetime(debt_cycle_df.Date)
debt_cycle_df = debt_cycle_df.set_index('Date')
debt_cycle_df = debt_cycle_df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1)
# print(debt_cycle_df.head())

debt_cycle_df['Close'] = debt_cycle_df['Close'].diff().dropna()
print(debt_cycle_df.head())

# split data
forecast_year = 54 # 마지막 forecast_year 년 예측,  월별데이터는 1년 예측하려면 12줘야함
df_train = debt_cycle_df.iloc[:-forecast_year]
df_test = debt_cycle_df.iloc[len(df_train):]


# from statsmodels.tsa.stattools import adfuller

# def adfuller_test(sales):
#     result=adfuller(sales)
#     labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
#     for value,label in zip(result,labels):
#         print(label+' : '+str(value) )
#     if result[1] <= 0.05:
#         print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
#     else:
#         print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# plt.plot( debt_cycle_df['Close'].diff().dropna(),label = "df의 사이클", color = 'r')
# plt.show()
# adfuller_test(debt_cycle_df['Close'].diff().dropna())







mlflow.set_experiment("New data ARIMA")
mlflow.autolog()

# p_values = [0, 1]
# d_values = [1, 2]
# q_values = [0, 1]

p_values = [1]
d_values = [0, 1, 2]
q_values = [1]

for p in p_values:
    for d in d_values:
        for q in q_values:
            # if d == 0:
            #     continue
             
            with mlflow.start_run() as run :
                df_train.index = pd.DatetimeIndex(df_train.index.values, freq=df_train.index.inferred_freq)
                model=ARIMA(df_train, order=(p, d, q))
                model_fit=model.fit() # 학습시키기
                print(model_fit.summary())
                # df['forecast_ARIMA']=model_fit.predict(start="2023-01-01",end="2198-01-01",dynamic=True)
                predictions_arima = model_fit.predict(start=len(df_train), end=len(debt_cycle_df))

                 # Evaluate the model using mean squared error
                mape_arima = np.mean(np.abs((df_test.Close - predictions_arima) / df_test.Close)) * 100


                mlflow.log_param("ARIMA_p", p)
                mlflow.log_param("ARIMA_d", d)
                mlflow.log_param("ARIMA_q", q)
                mlflow.log_metric("MAPE", mape_arima)

                
                # Save the model using MLflow's model logging capability
                plot_filename = "arima_predictions_plot.png"
                plt.plot(debt_cycle_df, label = "Actual", color = 'r')
                plt.plot(predictions_arima, label = "Predicted", color = 'b')
                plt.xlabel("Date")
                plt.ylabel("%")
                plt.title("ARIMA PREDICTION")
                plt.legend()
                plt.grid()
                plt.savefig(plot_filename)
                plt.clf()
                mlflow.log_artifact(plot_filename)

mlflow.end_run()

