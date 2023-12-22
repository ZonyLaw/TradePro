import os
import json
import numpy as np
import time

from .ig_service import IGService
from tickers.models import Ticker
from ..models import Price

if os.path.isfile('env.py'):
    import env


def recordIGPrice(data, data_next, scaling_factor):
    
    """
    This function is specifically designed for importing IG data into price DB.
    
    Args:
    data (dictionary): data containing the close, high, low, prices.
    data_next (dictionary): next set of data containing the close, high, low, prices; 
                            used for to get the next opening price for the next record.
    scaling_factor (integer): scale factor to turn the value into prices.
    """
    

    #setting objects of the DB
    ticker_instance = Ticker.objects.get(symbol="USDJPY")   
    new_price = Price()
    
    hour = new_price.date.hour
    
    #putting entries into price DB
    new_price.ticker = ticker_instance
    new_price.high = data['high'] / scaling_factor
    new_price.low = data['low'] / scaling_factor
    new_price.ask = data['offer'] / scaling_factor
    new_price.bid = data['bid'] / scaling_factor
   
    #calculate the open price
    new_price.close =  np.average([new_price.ask, new_price.bid]) 
    new_price.open_next = np.average([data_next['offer'], data_next['bid']]) / scaling_factor
    
    try:
        latest_price = Price.objects.filter(ticker=ticker_instance).latest()
        new_price.open = latest_price.open_next
        
    except:
        print("No open price found, so close price assigned!")
        new_price.open = new_price.close
        
    new_price.save()
    

def run_IG_mock():
    
    """
    This function executes a mock version to test the import of data into price DB.
    This is for development purpose.
    """
    
    print("Mock executed!")
    current_directory = os.getcwd()
    output_file_path = "price/data/IG_output.txt"
    file_path = os.path.abspath(os.path.join(current_directory, output_file_path))

    # Reading data from the file
    # Now 'data' is a Python dictionary containing the JSON data
    with open(file_path, 'r') as input_file:
        data = json.load(input_file)

    print(data['snapshot']['bid'])
    print(data['snapshot']['offer'])
    print(data['snapshot']['high'])
    print(data['snapshot']['low'])
    print(data['snapshot']['updateTime'])

    recordIGPrice(data['snapshot'], 100)


def run_IG(ticker):
    
    """
    This function execute the IG account to fetch prices for a ticker
    
    Args:
    ticker (string): ticker symbol (epic)
    
    """
    
    #this is to access the name of the ticker interested
    ticker_definition = {ticker:"CS.D.USDJPY.TODAY.IP"}
    
    #autheticating IG account and creating session    
    username = os.environ.get('IG_USERNAME')
    password = os.environ.get('IG_PASSWORD')
    api_key = os.environ.get('IG_API_KEY')
    acc_type = "LIVE"
    
    ig_service = IGService(username, password, api_key, acc_type)
    ig_service.create_session()
    
    try:
        #fetching data from IG account
        data = ig_service.fetch_market_by_epic(ticker_definition[ticker])
        
        #intial idea was to use prices from the next 10 second but this could lock up the api.
        #therefore, we will just use the current data set.
        #the purpose of this is to get the close price.
        # time.sleep(10)
        # data_next = ig_service.fetch_market_by_epic(ticker)
        
        recordIGPrice(data['snapshot'], data['snapshot'], 100)

    except:
        print("failed to retrieve data so running IG mock")
        run_IG_mock()