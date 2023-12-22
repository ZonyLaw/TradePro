import json
import os
# from .utils.predictive_analysis import model_run

def output_model_results():
    pred_reverse = {"key1":0.3, "key2":0.4}
    # parent_directory = r"C:\Users\sunny\Desktop\Development\python\TradePro"
    # pred_reverse_file = rf"{parent_directory}\pred_reverse.txt"
    
    pred_reverse_file = "/ml_models/temp/pred_reverse.txt"

    # pred_reverse, pred_continue = model_run()

    with open(pred_reverse_file, 'w') as file: 
        json.dumps(pred_reverse, file)
    
    print("completed", pred_reverse_file)
        
    # with open('pred_continue.txt', 'w') as convert_file: 
    #     convert_file.write(json.dumps(pred_continue))
    
    

def read_json_file(file_path):
    """
    Read data from a JSON file.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - data (dict): The data read from the JSON file.
    """
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")
        return None

# # Example usage:
# file_path = r"C:\Users\sunny\Desktop\Development\python\TradePro\pred_reverse.txt"
# result = read_json_file(file_path)

# if result is not None:
#     print("Data read from JSON file:")
#     print(result)
# else:
#     print("Error reading JSON file.")
    

output_model_results()