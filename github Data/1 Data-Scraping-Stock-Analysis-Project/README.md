# Data-Scraping-Stock-Analysis-Project
The idea of the project is to scrape NASDAQ website using selenium and analyze the stocks using python data libraries namely numpy and pandas
Datascrapper.py scrapes through the pages and stores the data into a csv file.
Alternatively we can use FastScrapper.py to download the data, it uses the download option present on the Nasdaq website. 
Run the StockScrapAndPlot.py which imports DataScrapper.py and StockAnalysisAndPlotting.py , the object first calls the scrape method in DataScrapper.py, the number of stocks and stock symbols are to be given by the user to which are then scrapped from the website. 
later and object is created to call different methods in StockAnalysisAndPlotting.py which wrangle and plot different aspects of stocks including closing prices , moving average, correlation,risk estimation and future stock behavior




