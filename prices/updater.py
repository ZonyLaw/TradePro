from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from .IG_session.run_IG2 import run_IG, run_IG_mock
from datetime import datetime, timedelta

# https://stackoverflow.com/questions/69387768/running-apscheduler-cron-at-certain-interval-combining-minutes-and-seconds


def start():
    print("starting scheduler")
    ticker = "USDJPY"
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_IG, 'date', args=[ticker], run_date=datetime.now() + timedelta(seconds=1))
   
    # scheduler.add_job(run_IG("USDJPY"), 'interval', args=[ticker_name], hours=1)
    scheduler.start()
    print("ending scheduler")
