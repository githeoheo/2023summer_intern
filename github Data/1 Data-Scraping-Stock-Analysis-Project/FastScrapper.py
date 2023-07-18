from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
import glob
import os
import numpy as np
import pandas as pd

class FastScrape:
    capab=DesiredCapabilities.CHROME
    capab["pageLoadStrategy"] = "eager"
    chrome_options = Options()

    no=int(input("Enter no of stocks:"))
    stocks=[]
    for i in range(no):
        stock=input("Enter Stock Symbol")
        stocks.append(stock)

    driver=Chrome("C:\\Users\\gopib\\Downloads\\chromedriver.exe", desired_capabilities=capab,
                            options=chrome_options)
    url="https://www.nasdaq.com/market-activity/quotes/historical"

    def fast_scrape(self):
        for stock in stocks:
            driver.get(url)
            driver.maximize_window()
            time.sleep(3)
            driver.find_element_by_id("find-symbol-input-dark").send_keys(stock,Keys.ENTER)
            if "No Results Found" in driver.page_source:
                print("Enter Correct Stock Symbol")
                quit()
            else:
                time.sleep(5)
                try:
                    driver.find_element_by_id("_evidon-decline-button").click()
                except:
                    # when doing multiple stocks cookie decline does not appear the second time, hence putting a try catch
                    pass
                driver.find_element_by_xpath(
                    "/html/body/div[2]/div/main/div[2]/div[4]/div/section/div/div[3]/ul/li[2]/a/span").click()
                time.sleep(5)
                driver.find_element_by_xpath("/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[1]/div[3]/div/div/div/button[6]").click()
                time.sleep(2)
                driver.find_element_by_xpath(
                    "/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[1]/div[3]/button").click()

        path="C:\\Users\\gopib\\Downloads\\Stocks"

        csv_files=glob.glob(os.path.join(path,"*.csv"))

        for file in csv_files:
            with open(file,'r') as data:
                text=data.read()
                text=text.replace("$","")
            with open(file,'w') as data1:
                data1.write(text)





