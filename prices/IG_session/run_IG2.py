import os
import json
import numpy as np
from datetime import datetime, timedelta
from django.utils import timezone

from .ig_service import IGService
from tickers.models import Ticker
from ..models import Price

if os.path.isfile('env.py'):
    import env


def recordIGPrice(ticker, df, scaling_factor):
    
    """
    This function is specifically designed for importing IG data into price DB.
    
    Args:
    df (dataframe)): df containing open, close, high, low prices as bid and ask
    scaling_factor (integer): scale factor to turn the value into prices.
    """

    #setting objects of the DB
    ticker_instance = Ticker.objects.get(symbol=ticker)   
    new_price = Price()
    
    #putting entries into price DB
    new_price.ticker = ticker_instance
        
    # Original timestamp
    original_timestamp_str = df['snapshotTime'][0]
    print(original_timestamp_str)

    date_str, time_str = original_timestamp_str.split('-')

    # Split the date components
    year, month, day = map(int, date_str.split(':'))

    # Split the time components
    hour, minute, second = map(int, time_str.split(':'))

    # Create a datetime object
    result_datetime = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    # manual_date = datetime(2023, 12, 22, 16, 00, 00, tzinfo=timezone.utc)

    #calculate the open price
    new_price.date = result_datetime
    new_price.open =  np.average([df['openPrice'][0]['ask'], df['openPrice'][0]['bid']]) / scaling_factor
    new_price.close =  np.average([df['closePrice'][0]['ask'], df['closePrice'][0]['bid']]) / scaling_factor
    new_price.high =  np.average([df['highPrice'][0]['ask'], df['highPrice'][0]['bid']]) / scaling_factor
    new_price.low =  np.average([df['lowPrice'][0]['ask'], df['lowPrice'][0]['bid']]) / scaling_factor
    new_price.volume = df['lastTradedVolume'][0]
    
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
    This function execute the IG account to fetch prices for a ticker using the history function.
    This IG fetch method provide open, close, high, low prices in addition trade volumn.
    
    Args:
    ticker (string): ticker symbol (epic)
    
    """
    
    #creating the time range for the fetch method
    # current_time = datetime.now()
    current_time_str = "2023-12-22 15:00:00"
    current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")
    next_hour = current_time + timedelta(hours=1)
    start_time_rounded = current_time.replace(minute=0, second=0, microsecond=0)
    end_time_rounded = next_hour.replace(minute=0, second=0, microsecond=0)
    target_date = start_time_rounded.strftime("%Y-%m-%d %H:%M:%S")
    start_time_str = start_time_rounded.strftime("%Y:%m:%d-%H:%M:%S")
    end_time_str = end_time_rounded.strftime("%Y:%m:%d-%H:%M:%S")
    

    # Check if prices exist for the target date
    # note the time format is different from the range date format
    prices_exist = Price.objects.filter(date=target_date).exists()
    print(prices_exist)
    
    if not prices_exist:
        
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
            print("get prices")
            data = ig_service.fetch_historical_prices_by_epic_and_date_range(ticker_definition[ticker], "HOUR",start_time_str, end_time_str )
            # data = ig_service.fetch_historical_prices_by_epic_and_date_range(ticker_definition[ticker], "HOUR","2023:12:22-15:00:00", "2023:12:22-17:00:00" )
            df = data['prices']
            recordIGPrice(ticker, df, 100)

        except:
            print("failed to retrieve data so running IG mock")
            # run_IG_mock()