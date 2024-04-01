import os
import sys
import datetime
import numpy as np
import pytz

from django.conf import settings
from datetime import datetime, timedelta
from django.utils import timezone

from .ig_service import IGService
from tickers.models import Ticker
from ..models import Price
from django.db.models import Count
from tradepro.utils.read_json import read_ticker_list
import logging

if os.path.isfile('env.py'):
    import env
    
#getting directory of the script.
base_dir = os.path.dirname(os.path.abspath(__file__))  
# Move up two levels from the current module's directory
base_dir_up_two_levels = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
relative_path = os.path.join(base_dir_up_two_levels, 'media', 'model_results', "logfile.log")
logging.basicConfig(filename=relative_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def recordIGPrice(ticker, df, scaling_factor):
    
    """
    This function is specifically designed for importing IG data into price DB.
    
    Args:
    ticker (string): this is ticker symbol
    df (dataframe)): df containing open, close, high, low prices as bid and ask
    scaling_factor (integer): scale factor to turn the value into prices.
    """

     # Setting objects of the DB
    ticker_instance = Ticker.objects.get(symbol=ticker)

    for _, row in df.iterrows():
        # Creating a new Price instance for each row
        new_price = Price()

        # Putting entries into price DB
        new_price.ticker = ticker_instance

        # Original timestamp
        original_timestamp_str = row['snapshotTime']
        print("Date for price recording:", original_timestamp_str)

        date_str, time_str = original_timestamp_str.split('-')

        # Split the date components
        year, month, day = map(int, date_str.split(':'))

        # Split the time components
        hour, minute, second = map(int, time_str.split(':'))

        # Create a datetime object
        result_datetime = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        print("the result datatime:", result_datetime)
        try:
        
            prices_exist = Price.objects.filter(ticker__symbol=ticker_instance.symbol, date=result_datetime).exists()
            
            print(prices_exist)
            if not prices_exist:      
                # Calculate the open price
                new_price.date = result_datetime
                new_price.open = np.average([row['openPrice']['ask'], row['openPrice']['bid']]) / scaling_factor
                new_price.close = np.average([row['closePrice']['ask'], row['closePrice']['bid']]) / scaling_factor
                new_price.high = np.average([row['highPrice']['ask'], row['highPrice']['bid']]) / scaling_factor
                new_price.low = np.average([row['lowPrice']['ask'], row['lowPrice']['bid']]) / scaling_factor
                new_price.volume = row['lastTradedVolume']

                new_price.save()
            else:
                price_instance = Price.objects.get(ticker__symbol=ticker_instance.symbol, date=result_datetime)
                print(price_instance)
                price_instance.open = np.average([row['openPrice']['ask'], row['openPrice']['bid']]) / scaling_factor
                price_instance.close = np.average([row['closePrice']['ask'], row['closePrice']['bid']]) / scaling_factor
                price_instance.high = np.average([row['highPrice']['ask'], row['highPrice']['bid']]) / scaling_factor
                price_instance.low = np.average([row['lowPrice']['ask'], row['lowPrice']['bid']]) / scaling_factor
                price_instance.volume = row['lastTradedVolume']
                price_instance.save()
                print("Already exist so no new entry created", result_datetime)
        except Exception as e:
            print("Fail to record the price")
            print(f"An unexpected error occurred: {e}")
            

    # Assuming 'date' is the field you want to check for duplicates
    duplicate_dates = Price.objects.values('date').annotate(count=Count('date')).filter(count__gt=1)
    print(duplicate_dates)
    for duplicate in duplicate_dates:
        # Keep one instance and delete the others
        duplicate_instances = Price.objects.filter(date=duplicate['date'])
        instance_to_keep = duplicate_instances.order_by('id').first()
        duplicate_instances.exclude(id=instance_to_keep.id).delete()


def run_IG(ticker, start_date=None, end_date=None):
    
    """
    This function execute the IG account to fetch prices for a ticker using the history function.
    This IG fetch method provide open, close, high, low prices in addition trade volumn.
    
    Args:
    ticker (string): ticker symbol (epic)
    
    """
    #this is to access the name of the ticker interested
    # ticker_definitions = {"USDJPY":"CS.D.USDJPY.TODAY.IP", "EURUSD":"CS.D.EURUSD.TODAY.IP",}
    ticker_definitions = read_ticker_list()['ticker_definitions']
    
    
    
    #creating the time range for the fetch method
    current_time = datetime.now()

    
    # current_time_str = "2023-12-29 14:50:00+00"
    # 2023-12-26 02:19:00+00:00

    #TODO: there is an issue how it deals with time so I subtract two hours below.
    if start_date is not None:
        print("Using range")
        start_range = start_date
        end_range = end_date
    else:
        print("Using cron auto get price")
        
        london_timezone = pytz.timezone('Europe/London')
        current_time_london = current_time.replace(tzinfo=pytz.utc).astimezone(london_timezone)
        current_time_str = current_time_london.strftime("%Y-%m-%d %H:%M:%S")
        
        start_range = current_time - timedelta(hours=1)
        end_range = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S") 
        print("using Cron start time", start_range)
        print("using Cron end time", end_range)
    
    start_time_rounded = start_range.replace( minute=0, second=0, microsecond=0)
    end_time_rounded = end_range.replace( minute=0, second=0, microsecond=0)
    start_time_str = start_time_rounded.strftime("%Y:%m:%d-%H:%M:%S")
    end_time_str = end_time_rounded.strftime("%Y:%m:%d-%H:%M:%S")
    print("rounded start time", start_time_rounded)
    print("rounded end time", end_time_rounded)
    logging.info(f"The input start time: {start_time_rounded}")
   
    #######Running the model predictions
    current_module_dir = os.path.dirname(os.path.abspath(__file__))
    ml_models_path = os.path.abspath(os.path.join(current_module_dir, '..', '..', 'ml_models','utils'))
    sys.path.append(ml_models_path)
    
    #autheticating IG account and creating session    
    username = os.environ.get('IG_USERNAME')
    password = os.environ.get('IG_PASSWORD')
    api_key = os.environ.get('IG_API_KEY')
    acc_type = "LIVE"
    
    ig_service = IGService(username, password, api_key, acc_type)
    ig_service.create_session()
    
    
    try:
        #fetching data from IG account
        print("Starting the fetch")
        data = ig_service.fetch_historical_prices_by_epic_and_date_range(f"{ticker_definitions[ticker]}", "HOUR",start_time_str, end_time_str )
        # data = ig_service.fetch_historical_prices_by_epic_and_date_range(ticker_definition[ticker], "HOUR","2023:12:25-12:00:00", "2023:12:25-13:00:00" )
        df = data['prices']
        print("This is df retrieved>>>", df)
        ig_service.logout()
        recordIGPrice(ticker, df, 100)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



