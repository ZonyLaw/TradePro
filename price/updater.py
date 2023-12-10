from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from .IG_session.run_IG import run_IG, run_IG_mock
from datetime import datetime, timedelta

# https://stackoverflow.com/questions/69387768/running-apscheduler-cron-at-certain-interval-combining-minutes-and-seconds


def start():
    print("starting scheduler")
    ticker_name = "CS.D.USDJPY.TODAY.IP"
    scheduler = BackgroundScheduler()
   # scheduler.add_job(checking, 'interval', seconds=30)
   # scheduler.add_job(get_price, 'cron', second=10)
    # scheduler.add_job(run_IG, 'cron', minute='*/3')
    # scheduler.add_job(run_IG, 'date', args=[ticker_name], run_date=datetime.now() + timedelta(seconds=1))
    # scheduler.start()
    print("ending scheduler")
