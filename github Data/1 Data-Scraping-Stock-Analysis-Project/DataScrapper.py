from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time
import string, re
import datetime

capab = DesiredCapabilities.CHROME
"""This will make Selenium WebDriver to wait until the initial HTML document has been completely loaded and parsed, 
and discards loading of stylesheets, images and sub-frames."""
capab["pageLoadStrategy"] = "eager"

"""The headless option will run everything in background. """
chrome_options = Options()
# chrome_options.add_argument("--headless")


class Scrapper:
    def __init__(self):
        self.url = "https://www.nasdaq.com/market-activity/quotes/historical"

    @staticmethod
    def stock_list():
        no_of_stocks = int(input("Enter no of stocks:"))
        stocks = []
        if no_of_stocks > 1:
            print("Enter Stock Symbols:")
        elif no_of_stocks < 1:
            print("No stocks entered, Quiting..")
            quit()
        elif no_of_stocks == 1:
            print("Enter Stock Symbol:")

        for i in range(no_of_stocks):
            stock = input()
            stocks.append(stock)
        return stocks

    def scrape(self):
        stocks = self.stock_list()
        driver = Chrome("C:\\Users\\gopib\\Downloads\\chromedriver.exe", desired_capabilities=capab,
                        options=chrome_options)

        for stock in stocks:
            driver.get(self.url)
            time.sleep(3)
            driver.maximize_window()
            # The below statement is for stopping the page load
            driver.execute_script("window.stop()")
            # We are finding the search box to enter the stock symbol
            driver.find_element_by_id(
                "find-symbol-input-dark").send_keys(
                str(stock), Keys.ENTER)

            if "No Results Found" in driver.page_source:
                print("Enter Correct Stock Symbol")
                quit()
            else:
                # To click the Historical quotes link, the cookie message which is appearing,
                # is blocking Historical Quotes from being interactable
                # So Clicking the "I Understand" Cookie button
                # time.sleep is for the cookie button to load
                # try catch block is needed since, when we are looping for multiple stocks, the cookie button does not appear the 2nd time and program will raise an exception

                time.sleep(5)
                try:
                    driver.find_element_by_id("_evidon-decline-button").click()
                except:
                    #when doing multiple stocks cookie decline does not appear the second time, hence putting a try catch
                    pass

                # The below is the Historical Quotes Links full Xpath
                driver.find_element_by_xpath(
                    "/html/body/div[2]/div/main/div[2]/div[4]/div/section/div/div[3]/ul/li[2]/a/span").click()

                # we can also use wait until clickable , i chose to use sleep
                time.sleep(5)

                # Click on 1 Year Tab
                driver.find_element_by_xpath("/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[1]/div[3]/div/div/div/button[4]").click()
                # element = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[1]/div[3]/div/div/div/button[4]")))
                # element.click()


                # Creating the file to save data
                file_columns = ["Date", "Close/Last", "Volume", "Open", "High", "Low"]
                with open(str(stock) + '.csv', 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(file_columns)
                    # Determining the number of times to loop by counting the number of buttons/pages in pagination
                    pagecount=1
                    while driver.find_element_by_class_name("pagination__next").is_enabled():
                        time.sleep(1)
                        try:
                            driver.find_element_by_class_name("pagination__next").click()
                            pagecount += 1
                        # if stock has only one page of data exception occurs in above
                        except:
                            break
                    time.sleep(3)
                    # Click on 1 Year Tab again to loop for data this time, we have created a speerate loop since for recent ipo stocks, which are only in one page , it is easier to handle with seperate loops,also there are some data duplication if we are using one loop
                    driver.find_element_by_xpath("/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[1]/div[3]/div/div/div/button[4]").click()
                    # the sleep below is to make sure that we are not reading from the last page from the above loop which we are using to detremine the number of times to loop
                    time.sleep(5)
                    # if page count is greater than 1 we iterate by clicking pagination_next
                    if pagecount>1:
                        for i in range(pagecount):
                            # the Tr table element that has all the rows
                            trs = driver.find_elements_by_xpath('/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[1]/div[4]/div[2]/div/table/tbody[2]')
                            # looping through the rows
                            for tr in trs:
                                data = str(tr.text).split()
                                # data above is stored as one chunck, we need to split data to 6 columns and add row by row
                                length = int(data.__len__() / 6)
                                j = 0
                                for i in range(length):
                                    data_row=[]
                                    data_row.append(data[0 + i + j].replace('/','-'))
                                    data_row.append(data[1 + i + j].replace('$', ''))
                                    data_row.append(data[2 + i + j].replace(',', ''))
                                    data_row.append(data[3 + i + j].replace('$', ''))
                                    data_row.append(data[4 + i + j].replace('$', ''))
                                    data_row.append(data[5 + i + j].replace('$', ''))
                                    writer.writerow(data_row)
                                    # the below is to fetch the 7th element and put it in the next row
                                    j += 5
                            time.sleep(2)
                            driver.find_element_by_class_name("pagination__next").click()
                           # sleep is added below to avoid reading the same page again
                            time.sleep(2)
                        csv_file.close()
                        print("No of scraped pages for",stock,"are",pagecount)
                    # if page count is less than 1 we read that page once
                    else:
                        trs = driver.find_elements_by_xpath('/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[1]/div[4]/div[2]/div/table/tbody[2]')
                        for tr in trs:
                            data = str(tr.text).split()
                            length=int(data.__len__()/6)
                            j=0
                            for i in range(length):
                                data_row = []
                                data_row.append(data[0 + i + j].replace('/', '-'))
                                data_row.append(data[1 + i + j].replace('$', ''))
                                data_row.append(data[2 + i + j].replace(',', ''))
                                data_row.append(data[3 + i + j].replace('$', ''))
                                data_row.append(data[4 + i + j].replace('$', ''))
                                data_row.append(data[5 + i + j].replace('$', ''))
                                writer.writerow(data_row)
                                j += 5

                        csv_file.close()
                        print("No of scraped pages for", stock, "are", pagecount)



