import os
import unittest
from unittest.mock import MagicMock, patch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prices.IG_session.ig_service import IGService  # Use absolute import

import env  # Import the env.py file

# Function to test
def connect_to_ig_account_and_create_session():
    # Retrieve environment variables
    username = os.environ.get('IG_USERNAME')
    password = os.environ.get('IG_PASSWORD')
    api_key = os.environ.get('IG_API_KEY')
    acc_type = "LIVE"
    
    # Create IGService instance and create session
    ig_service = IGService(username, password, api_key, acc_type)
    ig_service.create_session()

class TestConnectToIGAccount(unittest.TestCase):
    def test_connect_to_ig_account_and_create_session(self):
        # Call the function
        connect_to_ig_account_and_create_session()
        
if __name__ == '__main__':
    unittest.main()
