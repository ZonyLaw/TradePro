import os
import sys
import pandas as pd
import importlib.util
import datetime
import numpy as np

from django.conf import settings
from datetime import datetime, timedelta
from django.shortcuts import get_object_or_404
from django.utils import timezone

from .ig_service import IGService
from tickers.models import Ticker
from ..models import Price
from django.db.models import Count
from ml_models.utils.analysis_comments import compare_version_results
from ml_models.utils.access_results import read_prediction_from_json
from ml_models.utils.predictive_analysis import standard_analysis
from tradepro.utils.email import send_email

if os.path.isfile('env.py'):
    import env


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
    ticker_definition = {"USDJPY":"CS.D.USDJPY.TODAY.IP"}
    #creating the time range for the fetch method
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # current_time_str = "2023-12-29 14:50:00+00"
    # 2023-12-26 02:19:00+00:00

    if start_date is not None:
        print("Using range")
        start_range = start_date
        end_range = end_date
    else:
        print("Using cron auto get price")
        start_range = current_time - timedelta(hours=1)
        end_range = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")
    
    
    start_time_rounded = start_range.replace(minute=0, second=0, microsecond=0)
    end_time_rounded = end_range.replace(minute=0, second=0, microsecond=0)
    start_time_str = start_time_rounded.strftime("%Y:%m:%d-%H:%M:%S")
    end_time_str = end_time_rounded.strftime("%Y:%m:%d-%H:%M:%S")
    print(start_time_rounded)
    print(end_time_rounded)
   
    #######Running the model predictions
    current_module_dir = os.path.dirname(os.path.abspath(__file__))
    ml_models_path = os.path.abspath(os.path.join(current_module_dir, '..', '..', 'ml_models','utils'))
    sys.path.append(ml_models_path)
    
    standard_analysis("v4")
    standard_analysis("v5")
    standard_analysis("1h_v5")
    
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
        data = ig_service.fetch_historical_prices_by_epic_and_date_range(f"{ticker_definition[ticker]}", "HOUR",start_time_str, end_time_str )
        # data = ig_service.fetch_historical_prices_by_epic_and_date_range(ticker_definition[ticker], "HOUR","2023:12:25-12:00:00", "2023:12:25-13:00:00" )
        df = data['prices']
        print("This is df retrieved>>>", df)
        ig_service.logout()
        recordIGPrice(ticker, df, 100)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        
        
    ticker_instance = get_object_or_404(Ticker, symbol="USDJPY")
    prices = Price.objects.filter(ticker=ticker_instance)
    
    #sort prices table in ascending so latest price on the bottom
    #note that html likes to work with array if using indexing
    prices_df = pd.DataFrame(list(prices.values()))
    sorted_prices_df = prices_df.sort_values(by='date', ascending=True)
    last_four_prices_df = sorted_prices_df.tail(4)
        
    pred_reverse_v4 = read_prediction_from_json(f'USDJPY_pred_reverse_v4.json')
    pred_reverse_v5 = read_prediction_from_json(f'USDJPY_pred_reverse_v5.json')
    pred_reverse_1h_v5 = read_prediction_from_json(f'USDJPY_pred_reverse_1h_v5.json')
    
    version_comment, send_email_enabled = compare_version_results(pred_reverse_v4, pred_reverse_v5, pred_reverse_1h_v5, last_four_prices_df, 1 )
    
    check_email_alert(version_comment, send_email_enabled)


def check_email_alert(comment, send_email_enabled):
    print("here is the tag for email >>>>>>", send_email_enabled)
    current_day = datetime.now().weekday()
    current_time = datetime.now().time()
    if current_day in [5, 6]:
        print("It's the weekend. Email sending is disabled.")
    elif (1 <= current_time.minute <= 5 or 31 <= current_time.minute <= 35) and (send_email_enabled):
        try:
            send_email("sunny_law@hotmail.com", comment, "Alert-USDJPY potential trade")
        except Exception as e:
        # Catch specific exception types if possible, instead of a broad 'except' clause
            print(f"Error sending email: {e}")
