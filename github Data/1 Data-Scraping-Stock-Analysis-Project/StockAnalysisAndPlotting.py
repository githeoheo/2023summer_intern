import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import norm


class StockAnalysis:

    def __init__(self):
        pass

    def closing_prices(self):
        # path=os.getcwd()
        # print(path)
        # csv_files=glob.glob(os.path.join(path,"*.csv"))
        csv_files = glob.glob("*.csv")

        df_dict={}
        for file in csv_files:
            stock_name=file.replace('.csv','').upper()
            df_dict[stock_name] = pd.read_csv(file)
            df_dict[stock_name]['Date']=df_dict[stock_name]['Date'].astype('datetime64[ns]')
            df_dict[stock_name] = df_dict[stock_name].rename(columns={'Close/Last': 'Close'})
            df_dict[stock_name]['Close'] = df_dict[stock_name]['Close'].astype('float')
            df_dict[stock_name]['Volume'] = df_dict[stock_name]['Volume'].astype('int64')
            df_dict[stock_name]['Open'] = df_dict[stock_name]['Open'].astype('float')
            df_dict[stock_name]['High'] = df_dict[stock_name]['High'].astype('float')
            df_dict[stock_name]['Low'] = df_dict[stock_name]['Low'].astype('float')
            # print(df.dtypes)
            # print(df.info())
            df_dict[stock_name]=df_dict[stock_name].set_index('Date')
            # not using plt.figure(figsize=(20,10)) as it will create new plot for each stock in loop , we want only one plot
            plt.title('Closing Prices Trend for stocks', fontsize=16, fontname="Times New Roman")
            plt.xlabel('Date', fontsize=14, fontname="Times New Roman")
            plt.ylabel('Closing Price', fontsize=14, fontname="Times New Roman")
            # plt.xticks(rotation="vertical")
            plt.xticks(fontsize=14, fontname="Times New Roman")
            plt.yticks(fontsize=14, fontname="Times New Roman")
            df_dict[stock_name]['Close'].plot(figsize=(15,10))
            plt.legend(df_dict.keys())

        plt.savefig('Closing Prices')
        plt.close()

    def close_vs_moving_average(self):
        csv_files = glob.glob("*.csv")
        days = int(input("Enter first set of Moving Average No of Days to plot for :"))
        days2 = int(input("Enter second set of  Moving Average No of Days to plot for :"))
        days3 = int(input("Enter third set of  Moving Average No of Days to plot for :"))
        for file in csv_files:
            stock_name = file.replace('.csv', '').upper()
            df=pd.read_csv(file)
            df=df.rename(columns={'Close/Last':'Close'})
            df['Date'] = df['Date'].astype('datetime64[ns]')
            df['Close'] = df['Close'].astype('float')
            df['Volume'] = df['Volume'].astype('int64')
            df['Open'] = df['Open'].astype('float')
            df['High'] = df['High'].astype('float')
            df['Low'] = df['Low'].astype('float')
            df=df.set_index('Date')
            df[str(days)+' day moving_avg']=df['Close'].rolling(days).mean()
            df[str(days2) + ' day moving_avg'] = df['Close'].rolling(days2).mean()
            df[str(days3) + ' day moving_avg'] = df['Close'].rolling(days3).mean()
            df[['Close',str(days)+' day moving_avg',str(days2) + ' day moving_avg',str(days3) + ' day moving_avg']].plot(legend=True,figsize=(15,10))
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            title="Closing Price and Moving Average of "+str(days)+","+str(days2)+","+str(days3)+" days for "+str(stock_name)
            plt.title(title)
            filename = 'ClosingVsMoving_' + str(days) +"_"+str(days2)+"_"+str(days3)+"_" +str(stock_name)
            plt.savefig(filename)
            plt.close()

    def daily_returns_combined(self):
        # daily_rets = df.copy()
        # print(df[1:])
        # print(df[:-1])
        # daily_rets[1:] = (df[1:] / df[:-1].values) - 1
        # print(daily_rets)
        # daily_rets.iloc[0] = 0
        # daily_rets = daily_rets[1:]
        # print(daily_rets['Close'])

        csv_files = glob.glob("*.csv")
        df_dict={}
        for file in csv_files:
            stock_name=file.replace(".csv","").upper()
            df_dict[stock_name]=pd.read_csv(file)
            df_dict[stock_name]['Date']=df_dict[stock_name]['Date'].astype('datetime64[ns]')
            df_dict[stock_name] = df_dict[stock_name].rename(columns={"Close/Last": "Close"})
            df_dict[stock_name]['Close'] = df_dict[stock_name]['Close'].astype('float')
            df_dict[stock_name]['Volume'] = df_dict[stock_name]['Volume'].astype('int64')
            df_dict[stock_name]['Open'] = df_dict[stock_name]['Open'].astype('float')
            df_dict[stock_name]['High'] = df_dict[stock_name]['High'].astype('float')
            df_dict[stock_name]['Low'] = df_dict[stock_name]['Low'].astype('float')
            df_dict[stock_name]=df_dict[stock_name].set_index('Date')
            df_dict[stock_name]['daily_return'] = df_dict[stock_name]['Close'].pct_change()
            df_dict[stock_name]['daily_return'].plot(figsize=(20,10))
            # we can also plot distribution of daily returns using sns.distplot
            #plt.ylabel('Probability Density Value')
            # sns.distplot(df_dict['daily_return'].dropna(), bins = 100, color = 'red') #
            plt.xlabel('Date')
            plt.ylabel('Daily Returns')
            plt.title('Daily Returns for stocks combined plot')

        plt.legend(df_dict.keys())
        plt.savefig('Daily Returns of stocks combined')
        plt.close()

    def daily_returns_individual(self):
        csv_files=glob.glob("*.csv")
        for file in csv_files:
            stock_name=file.replace(".csv","").upper()
            df=pd.read_csv(file)
            df['Date']=df['Date'].astype('datetime64[ns]')
            df=df.set_index('Date')
            df=df.rename(columns={'Close/Last':'Close'})
            df['Close'] = df['Close'].astype('float')
            df['Volume'] = df['Volume'].astype('int64')
            df['Open'] = df['Open'].astype('float')
            df['High'] = df['High'].astype('float')
            df['Low'] = df['Low'].astype('float')
            df_daily=df.copy()
            df_daily[1:]=(df_daily[1:].values/df_daily[:-1].values)-1
            df_daily=df_daily.rename(columns={'Close':'Daily Returns'})
            df_daily=df_daily[1:]
            plt.figure(figsize=(20,10))
            plt.title('Daily Returns for '+stock_name)
            plt.xlabel('Date')
            plt.ylabel('Daily Returns')
            df_daily['Daily Returns'].plot()
            # for legend always surround by square brackets to print full name, sles it will show only first letter in legend
            plt.legend([stock_name])
            plt.savefig('Daily Returns of '+stock_name)
            plt.close()

    def correlation_close(self):
        csv_files = glob.glob("*.csv")
        df1 = pd.DataFrame()
        # df2=pd.DataFrame()
        #creating a string to get stock names for our plot title
        stocks=''
        for file in csv_files:
            stock_name = file.replace('.csv', '').upper()
            df = pd.read_csv(file)
            df = df.drop(columns={'Volume', 'Open', 'High', 'Low'})
            df = df.rename(columns={'Close/Last': 'Close'+"_"+stock_name})
            df['Date'] = df['Date'].astype('datetime64[ns]')
            df = df.set_index('Date')
            df1=df1.join(df,how="outer")
            # we can also use merge, merge gives more options than a join , we can join on columns too with merge unlike join which is only on index
            # df2=df2.merge(df,how="outer",left_index=True,right_index=True)
            stocks=stocks+stock_name+","

        # removing the last trailing comma
        stocks=stocks[:-1]
        corr = df1.corr()
        plt.figure(figsize=(20, 10))
        sns.heatmap(corr, annot=True)
        plt.title('Correlation of closing prices between stocks '+stocks)
        plt.savefig('CorrelationOfStocksClose')
        plt.close()


    def correlation_daily_returns(self):
        csv_files = glob.glob("*.csv")
        df1 = pd.DataFrame()
        # df2=pd.DataFrame()
        #creating a string to get stock names for our plot title
        stocks=''
        for file in csv_files:
            stock_name = file.replace('.csv', '').upper()
            df = pd.read_csv(file)
            df = df.drop(columns={'Volume', 'Open', 'High', 'Low'})
            df = df.rename(columns={'Close/Last':'Close'})
            df['Date'] = df['Date'].astype('datetime64[ns]')
            df = df.set_index('Date')
            df['Close']=df['Close'].astype(float)
            df['Daily Returns'+stock_name]=df['Close'].pct_change()
            df=df.drop(columns={'Close'})
            df1=df1.join(df,how="outer")
            # we can also use merge, merge gives more options than a join , we can join on columns too with merge unlike join which is only on index
            # df2=df2.merge(df,how="outer",left_index=True,right_index=True)
            stocks=stocks+stock_name+","

        # removing the last trailing comma
        stocks=stocks[:-1]
        corr = df1.corr()
        plt.figure(figsize=(20,10))
        sns.heatmap(corr, annot=True)
        plt.title('Correlation of daily returns between stocks '+stocks)
        plt.savefig('CorrelationOfStocksDailyReturns')
        plt.close()

    def risk_estimation(self):
        csv_files = glob.glob("*.csv")
        df_dict={}
        i=1
        # creating a string to get stock names for our plot title
        for file in csv_files:
            stock_name = file.replace('.csv', '').upper()
            df_dict[stock_name] = pd.read_csv(file)
            df_dict[stock_name]['Date'] = df_dict[stock_name]['Date'].astype('datetime64[ns]')
            df_dict[stock_name] = df_dict[stock_name].rename(columns={"Close/Last": "Close"})
            df_dict[stock_name]['Close'] = df_dict[stock_name]['Close'].astype('float')
            df_dict[stock_name]['Volume'] = df_dict[stock_name]['Volume'].astype('int64')
            df_dict[stock_name]['Open'] = df_dict[stock_name]['Open'].astype('float')
            df_dict[stock_name]['High'] = df_dict[stock_name]['High'].astype('float')
            df_dict[stock_name]['Low'] = df_dict[stock_name]['Low'].astype('float')
            df_dict[stock_name] = df_dict[stock_name].set_index('Date')
            df_dict[stock_name]['avg_daily_return'] = df_dict[stock_name]['Close'].pct_change().mean()
            df_dict[stock_name]['daily_risk'] = df_dict[stock_name]['Close'].pct_change().std()
            if i==1:
                plt.figure(figsize=(20,10))
                i+=1
            else:
                pass
            plt.xlabel('Daily Average Expected Return')
            plt.ylabel('Daily Risk')
            # s is the size parameter of the dot in the plot
            plt.scatter(df_dict[stock_name]['avg_daily_return'],df_dict[stock_name]['daily_risk'],s=30)

            # loss calculation, 0.1 gives 90% accuracy
            # Each number signifies that the total loss for a single day will not exceed this value
            investment = 100000
            loss = (abs(df_dict[stock_name]['avg_daily_return'].quantile(0.1))) * investment
            print(stock_name,"max loss can be ",loss,"for an investment of ",investment, " in a day with an accuracy of 90%")


        plt.title('Avg Daily Returns vs Risk of Stocks')
        plt.legend(df_dict.keys())
        plt.savefig('RiskvsReturnofStocks')

    def future_stock_behavior(self):
        csv_files=glob.glob("*.csv")
        df_dict = {}
        i = 1
        for file in csv_files:
            stock_name = file.replace('.csv', '').upper()
            df_dict[stock_name] = pd.read_csv(file)
            df_dict[stock_name]['Date'] = df_dict[stock_name]['Date'].astype('datetime64[ns]')
            df_dict[stock_name] = df_dict[stock_name].rename(columns={"Close/Last": "Close"})
            df_dict[stock_name]['Close'] = df_dict[stock_name]['Close'].astype('float')
            df_dict[stock_name] = df_dict[stock_name].drop(columns={'Volume', 'Open', 'High', 'Low'})
            df_dict[stock_name] = df_dict[stock_name].set_index('Date')
            df_dict[stock_name]['daily_return'] = df_dict[stock_name]['Close'].pct_change()
            if i == 1:
                plt.figure(figsize=(20, 10))
                i += 1
            else:
                pass
            log_returns = np.log(1 +  df_dict[stock_name]['daily_return'])  # Calculating log returns from daily returns

            avg = log_returns.mean()  # Calculating average of log returns
            var = log_returns.var()  # Calculating variance
            drift = avg - (var / 2.0)  # Calculating drift
            drift = np.array(drift)  # Convert to array

            pred_price_overDays = 60  # Number of days
            pred_count = 10  # Range of prediction

            std = log_returns.std()  # Calculating STD
            std = np.array(std)  # Convert to array

            x = np.random.rand(pred_price_overDays, pred_count)  # get random multidimensional array

            Rv = std * norm.ppf(x)  # Calculating Rv

            # print("The required Rv array is:\n", Rv)
            e_value = np.exp(drift + Rv)  # Calculating the E value

            current_price = df_dict[stock_name]['Close'].iloc[-1]  # Selecting last price of the year

            new_prices = np.zeros_like(e_value)  # create array to store the results

            new_prices[0] = current_price

            # print(new_prices)

            for i in range(1, pred_price_overDays):  # Loop over all the days to find their prices
                new_prices[i] = new_prices[i - 1] * e_value[i]  # Calculating the future price with formula

            # print("The Minimum Predicted Price:", new_prices[pred_price_overDays - 1].min())  # Get minimum price
            # print("The Maximum Predicted Price:", new_prices[pred_price_overDays - 1].max())  # Get maximum price

            plt.figure(figsize=(20,10))
            plt.xlabel('Days',fontsize=15)  # Assign name to x-axis
            plt.ylabel('Price',fontsize=15)  # Assign name to y-axis
            plt.title('Monte Carlo Analysis for '+ stock_name)  # Assign name to the plot
            plt.plot(new_prices)  # plot the figure
            plt.savefig(stock_name+" Future Prediction")
            plt.legend([stock_name])

            # print("\nThe price array:\n", new_prices)



f=StockAnalysis()
f.future_stock_behavior()