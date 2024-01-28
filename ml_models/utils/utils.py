import csv
from io import StringIO

def trade_direction(trade_diff):
    """_summary_
    This function determines the direction of the candle stick for any timeframe

    Args:
        trade_diff (float): the difference between open and current price.

    Returns:
        string: the trade direction of the candle stick
    """
    
    if trade_diff == 0:
        return "Neutral / Doji"
    elif trade_diff > 0:
        return "Buy"
    else:
        return "Sell"
    
def export_results(df):
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)

    # Write header
    csv_writer.writerow(['date', 'ticker', 'open', 'close', 'high', 'low'])

    # Write data
    for price in prices:
        csv_writer.writerow([price.date, price.ticker, price.open, price.close, price.high, price.low])

    # Get CSV data as a string
    csv_data = csv_buffer.getvalue()

    # Close the buffer
    csv_buffer.close()

    return csv_data

def comment_model_results(model_results_dict, model_results_label):
    array = model_results_dict[model_results_label][1]['item']['Potential Trade']
    current_trade = array[0]
    if "Buy" in current_trade:
        trade = "Sell"
    else:
        trade = "Buy"
        
    if trade in array:
        print(trade)
        comment = f"Scenario {model_results_label} - indicates there could be oscilation pattern or near-term retracement!"
    else:
        comment = f"Scenario {model_results_label} - indicates direction continues in {trade}!"
        
        
    return comment


def compare_version_results(model_dict1, model_dict2, model_dict3):
    key_label1 = list(model_dict1.keys())[0]
    array1 = model_dict1[key_label1][1]['item']['Potential Trade']
    current_trade1 = array1[0][:3]
    
    key_label2 = list(model_dict2.keys())[0]
    array2 = model_dict2[key_label2][1]['item']['Potential Trade']
    current_trade2 = array2[0][:3]
    
    key_label3 = list(model_dict3.keys())[0]
    array3 = model_dict2[key_label3][1]['item']['Potential Trade']
    current_trade3 = array3[0][:3]
   
    if current_trade1 == current_trade2 == current_trade3:
        comment = f"All model versions predict a {current_trade1}"
    elif current_trade2 == current_trade3:
        comment = f"4hr model and 1hr model predict the same {current_trade1}"
    else:
        comment = f"Unreliable predictions"
    
    return comment