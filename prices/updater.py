from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from .IG_session.run_IG2 import run_IG, run_IG_mock

# https://stackoverflow.com/questions/69387768/running-apscheduler-cron-at-certain-interval-combining-minutes-and-seconds


def start():
    print("starting scheduler")
    ticker = "USDJPY"
    print(type(ticker))
    scheduler = BackgroundScheduler()
   
    current_time = datetime.now()
    print("updater date and time", current_time)
    
    # Method 1: call the scheduler when app is loaded
    # scheduler.add_job(run_IG, 'date', args=[ticker], run_date=datetime.now() + timedelta(seconds=1))
    
    # Method 2: create the scheduler with adjusted time specified by next_hour
    # next_hour specify the minutes and seconds
    # next_hour = (current_time + timedelta(hours=1)).replace(minute=34, second=0, microsecond=0)
    next_hour = (current_time).replace(minute=1, second=0, microsecond=0)
    scheduler.add_job(run_IG, 'interval', args=[ticker], minutes=30, start_date=next_hour)
    # scheduler.add_job(run_IG, 'interval', args=[ticker], hours=1, start_date=next_hour)
    
    # these are extra versions but can be deleted later
    # scheduler.add_job(run_IG, 'interval', args=[ticker], hours=1)
    # scheduler.add_job(run_IG, 'cron', args=[ticker], hour='*')
    
    scheduler.start()
    print("ending scheduler")
