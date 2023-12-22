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
    # original_datetime = datetime.strptime(original_timestamp_str, "%Y:%m:%d-%H:%M:%S")

    # # Create a fixed offset for +00:00
    # fixed_offset = timedelta(hours=0)

    # # Convert to the desired format with microseconds and timezone information
    # formatted_timestamp = original_datetime.replace(microsecond=632928, tzinfo=fixed_offset)

    # # Format the datetime object as a string
    # formatted_timestamp_str = formatted_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f%z")

    date_str, time_str = original_timestamp_str.split('-')

    # Split the date components
    year, month, day = map(int, date_str.split(':'))

    # Split the time components
    hour, minute, second = map(int, time_str.split(':'))

    # print(minute, second)
    # Create a datetime object
    result_datetime = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    # manual_date = datetime(2023, 12, 22, 16, 00, 00, tzinfo=timezone.utc)

    print(result_datetime)
    
    
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
        
        data = ig_service.fetch_historical_prices_by_epic_and_date_range(ticker_definition[ticker], "HOUR","2023:12:22-14:00:00", "2023:12:22-15:00:00" )
        df = data['prices']
        recordIGPrice(ticker, df, 100)

    except:
        print("failed to retrieve data so running IG mock")
        # run_IG_mock()