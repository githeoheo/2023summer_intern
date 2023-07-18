from DataScrapper import *
from StockAnalysisAndPlotting import *
# from FastScrapper import *

if __name__=="__main__":
    obj=Scrapper()
    obj.scrape()
    # obj_f=FastScrape()
    # obj_f.fast_scrape()
    obj1 = StockAnalysis()
    obj1.closing_prices()
    obj1.close_vs_moving_average()
    obj1.daily_returns_combined()
    obj1.daily_returns_individual()
    obj1.correlation_close()
    obj1.correlation_daily_returns()
    obj1.risk_estimation()
    obj1.future_stock_behavior()



