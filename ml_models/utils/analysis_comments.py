import os
import sys
import importlib.util
import datetime

    
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
        comment = f"Scenario {model_results_label} - indicates consistent {current_trade.split()[0]} direction up to 40pips!"
        
        
    return comment


def compare_version_results(model_dict1, model_dict2, model_dict3):
    
    
    
    # Dynamically get the module path involves defining the parent directory
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Add the parent directory to the Python path
    sys.path.append(parent_directory)
    module_name = 'tradepro.utils.email'

    try:
        email = importlib.import_module(module_name)
    except ImportError:
        print("Error importing email module!")
        email = None
    
    key_label1 = list(model_dict1.keys())[0]
    array1 = model_dict1[key_label1][1]['item']['Potential Trade']
    current_trade1 = array1[0].split()[0]
    
    key_label2 = list(model_dict2.keys())[0]
    array2 = model_dict2[key_label2][1]['item']['Potential Trade']
    current_trade2 = array2[0].split()[0]
    
    key_label3 = list(model_dict3.keys())[0]
    array3 = model_dict3[key_label3][1]['item']['Potential Trade']
    current_trade3 = array3[0].split()[0]
   
    if current_trade1 == current_trade2 == current_trade3:
        comment = f"All model versions predict a {current_trade1}"
        
        current_day = datetime.datetime.now().weekday()
        if current_day in [5, 6]:
            print("It's the weekend. Email sending is disabled.")
        else:
            try:
                email.send_email("sunny_law@hotmail.com", comment, "Alert-USDJPY potential trade")
            except Exception as e:
            # Catch specific exception types if possible, instead of a broad 'except' clause
                print(f"Error sending email: {e}")
            
    elif current_trade2 == current_trade3:
        comment = f"4hr model and 1hr model predict the same {current_trade2}"
    else:
        comment = f"WARNING: Model predictions are inconsistant so not recommended for use!"
           
    return comment