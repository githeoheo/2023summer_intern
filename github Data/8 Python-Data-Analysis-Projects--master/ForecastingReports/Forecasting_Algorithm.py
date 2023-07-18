#!/bin/python
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import fbprophet

#   ------Build 1.0.0.0: Forecasting Algorithm Complete

class Forecast():
	def __init__(self, file):
		self.file = file

	def data_manipulation(self):
		ticker = pd.read_csv(self.file)
		date = ticker["Date"]
		close = ["Close"]

	def line_graph(self):
		plt.plot(date, close)
		plt.xlabel("Date")
		plt.ylabel("Close")
		plt.show()

	def forecast(self):
		prophet = fbprophet.Prophet()
		ticker[['ds', 'y']] = ticker[["Date", "Close"]]

		prophet.fit(ticker)
		prediction = prophet.make_future_dataframe(periods = 365)
		predict = prophet.predict(prediction)

		prophet.plot(predict)

if __name__ == "__main__":
	pass