from ig_service import IGService
import os

if os.path.isfile('env.py'):
    import env


def run_IG():
           
    username = os.environ.get('IG_USERNAME')
    password = os.environ.get('IG_PASSWORD')
    api_key = os.environ.get('IG_API_KEY')
    acc_type = "LIVE"
    
    ig_service = IGService(username, password, api_key, acc_type)
    ig_service.create_session()