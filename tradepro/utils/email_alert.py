from django.conf import settings
from datetime import datetime
from .email import send_email
from ml_models.utils.predictive_analysis import run_model_predictions


def email_freq_controller(freq_enabler):
    """
    This checks the condition to send email which limits sending at the hour and half an hour.
    Args:
    freq_enabler(boolean): if 1 then email will be condition to be sent only around 00 and 30 minutes;
        otherwise it will send despite of time
    
    """
    
    if freq_enabler:
        current_time = datetime.now().time()
        return 1 <= current_time.minute <= 5 or 31 <= current_time.minute <= 35
    else:
        return 1


def email_alert( model_ticker, comment, general_ticker_info, send_email_enabled, email_freq_enabler):
    """
    This function send email alert of the possible trade opportunities.
    This calls on the standard_analysis and compare_version_results functions.

    Args:
        model_ticker (string): specify the ticker to run predictions and email alert.
        email_freq_enabler (integer): This is boolean where 1 limits sending to certain time 
            and 0 allowing to happen any time. 
    """

    print("here is the tag for email >>>>>>", send_email_enabled)
    current_day = datetime.now().weekday()
    email_freq_condition = email_freq_controller(email_freq_enabler)
    
    if current_day in [5, 6]:
        print("It's the weekend. Email sending is disabled.")
    elif (email_freq_condition) and (send_email_enabled):
        try:
            send_email("sunny_law@hotmail.com", comment + general_ticker_info,
                       f"Alert-{model_ticker} potential trade")
        except Exception as e:
        # Catch specific exception types
            print(f"Error sending email: {e}")