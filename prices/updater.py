from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from django.conf import settings
from prices.IG_session.run_IG import run_IG
from tradepro.utils.email_alert import email_alert
from ml_models.utils.predictive_analysis import run_model_predictions
from tradepro.utils.read_json import read_ticker_list
# https://stackoverflow.com/questions/69387768/running-apscheduler-cron-at-certain-interval-combining-minutes-and-seconds


import logging
import os

current_directory = os.getcwd()
logging.basicConfig(filename=f'{current_directory}/media/logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start():
    logging.info("Starting scheduler")
    print("starting scheduler")
    ticker = "USDJPY"
    scheduler = BackgroundScheduler()
   
    current_time = datetime.now()
    print("updater date and time", current_time)
    
     
    logging.info("Updater date and time: %s", current_time)
    # Method 1: call the scheduler when app is loaded
    # scheduler.add_job(run_procedure_sequence, 'date', run_date=datetime.now() + timedelta(seconds=1))

        
    # Method 2: create the scheduler with adjusted time specified by next_hour
    # next_hour specify the minutes and seconds
    # next_hour = (current_time + timedelta(hours=1)).replace(minute=34, second=0, microsecond=0)

    next_hour = (current_time).replace(minute=1, second=0, microsecond=0)
    scheduler.add_job(run_procedure_sequence, 'interval', minutes=5, start_date=next_hour)
    
    # scheduler.add_job(run_IG, 'interval', args=[ticker], hours=1, start_date=next_hour)
    
    # these are extra versions but can be deleted later
    # scheduler.add_job(run_IG, 'interval', args=[ticker], hours=1)
    # scheduler.add_job(run_IG, 'cron', args=[ticker], hour='*')

    
    scheduler.start()
    print("ending scheduler")
    logging.info("Ending scheduler")


def run_procedure_sequence():
    logging.info("Inside the procedure sequence")
    print("inside the procedure sequence!!!!")
    # tickers = ["USDJPY","EURUSD"]
    #defines the list of tickers
    tickers = read_ticker_list()['tickers']
    
    for ticker in tickers:
        #update the price table
        run_IG(ticker)
        comparison_comment, general_ticker_info, send_email_enabled = run_model_predictions(ticker)
        # variability_analysis(ticker)
        email_alert(ticker, comparison_comment, general_ticker_info, send_email_enabled,1)
