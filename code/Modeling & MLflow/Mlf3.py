## MLflow 를 ARIMA에 적용해보기
## 데이터셋 : 부채사이클( debt_cycle ) 
## 깃허브 : https://github.com/krishnaik06/ARIMA-And-Seasonal-ARIMA/tree/master
## ARIMA-And-Seasonal-ARIMA

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

dataset_filepath = 'C:/Users/user/Desktop/Securities_Data_Analysis(Junsu)/dataset/original_data/debt_cycle.csv'
df=pd.read_csv(dataset_filepath)
# print(df.head())

# Convert Month into Datetime
df['Date']=pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)

# df['Cycle'].diff().dropna().plot()

plt.plot( df['Cycle'].diff().dropna(),label = "df의 사이클", color = 'r')
# plt.show()


from statsmodels.tsa.stattools import adfuller

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

# adfuller_test(df['Cycle'].diff().dropna())



import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Cycle'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Cycle'].iloc[13:],lags=40,ax=ax2)
# fig.show()
# plt.show()



import mlflow
# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
mlflow.set_experiment("ARIMA333")
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
            with mlflow.start_run() as run :
                df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)
                model=ARIMA(df['Cycle'].diff().dropna(), order=(p, d, q))
                model_fit=model.fit() # 학습시키기
                df['forecast_ARIMA']=model_fit.predict(start="2023-01-01",end="2198-01-01",dynamic=True)

                mlflow.log_param("ARIMA_p", p)
                mlflow.log_param("ARIMA_d", d)
                mlflow.log_param("ARIMA_q", q)
                # mlflow.log_metric("forecast", df['forecast_ARIMA'])

                # 여기 년도는 years=x  range 범위는 1 = 1년
                # 여기 월별로는 months=x  range 범위는 1 = 1달
                from pandas.tseries.offsets import DateOffset
                future_dates_arima=[df.index[-1]+ DateOffset(years=x) for x in range(0,75)]  # 20년
                # future_dates_arima=[df.index[-1]+ DateOffset(months=x) for x in range(0,36)]  # 3년
                future_datest_df_arima=pd.DataFrame(index=future_dates_arima[1:],columns=df.columns)
                future_df_arima=pd.concat([df,future_datest_df_arima])
                future_df_arima['forecast_ARIMA'] = model_fit.predict(start = "2023-01-01", end = "2123-01-01", dynamic= True)
                future_df_arima[['Cycle', 'forecast_ARIMA']].plot(figsize=(12, 8))
                # plt.show()            
                plot_filename = "arima_predictions_plot.png"
                plt.savefig(plot_filename)
                mlflow.log_artifact(plot_filename)




mlflow.end_run()