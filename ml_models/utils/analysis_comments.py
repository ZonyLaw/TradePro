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


def compare_version_results(model_dict1, model_dict2, model_dict3, prices_df, arr_index, newline_syntax):
    """
    The function provides comments based on consistency of models output. 

    Args:
        model_dict1 (dictionary): results from first model
        model_dict2 (dictionary): results from the second model
        model_dict3 (dictionary): results from the third model
        prices_df (dataframe): a list of prices in dataframe
        arr_index (integer): a index in the array to extract model results for comparison - refer to the json file
        newline_syntax (integer): 1 for '\n' otherwise <br> 

    Returns:
        string: analysis comments
    """
    
    key_label1 = list(model_dict1.keys())[0]
    array1 = model_dict1[key_label1][1]['item']['Potential Trade']
    current_trade1 = array1[arr_index].split()[0]
    
    key_label2 = list(model_dict2.keys())[0]
    array2 = model_dict2[key_label2][1]['item']['Potential Trade']
    current_trade2 = array2[arr_index].split()[0]
    
    key_label3 = list(model_dict3.keys())[0]
    array3 = model_dict3[key_label3][1]['item']['Potential Trade']
    current_trade3 = array3[arr_index].split()[0]
    
    date = prices_df.iloc[3]['date']
    open_price_1hr = round(prices_df.iloc[3]['open'], 2)
    open_price_4hr = round(prices_df.iloc[0]['open'], 2)
    close_price_1hr = round(prices_df.iloc[3]['close'], 2)
    entry_price_avg_1hr_4hr = round((prices_df.iloc[3]['open'] + prices_df.iloc[0]['open']) / 2, 2)
    
    
    if newline_syntax:
        next_line = "\n"
    else:
        next_line = "<br>"
    
    market_comments = (
            f"Market Information:{next_line}"
            f"Date: {date}{next_line}"
            f"The current open price is {open_price_1hr}.{next_line}"
            f"The current close price is {close_price_1hr}.{next_line}"
            f"The price 4 hours ago is {open_price_4hr}.{next_line}"
            f"The average entry price 4 hours ago is {entry_price_avg_1hr_4hr}"
    )
    
    send_email = 0
    if current_trade1 == current_trade2 == current_trade3:
        comment = (
            f"All model versions predict a {current_trade1}{next_line}"
            f"{market_comments}.{next_line}"
        )
        send_email = 1
                
    elif current_trade1 == current_trade2:
        comment = (
            f"Both 4hr models predict the same {current_trade2}{next_line}"
            f"{market_comments}.{next_line}"
        )
        send_email = 1
    
    elif current_trade2 == current_trade3:
        comment = (
            f"Oscillation possible: 4hr model and 1hr model predict the same {current_trade2}.{next_line}"
            f"{market_comments}.{next_line}"
        )
        send_email = 0
        
    elif current_trade1 == current_trade3:
        comment = (
            f"Oscillation possible: 4hr LAGGED model and 1hr model predict the same {current_trade2}{next_line}"
            f"{market_comments}.{next_line}"
        )
        send_email = 0
        
    else:
        comment = (
            f"WARNING: Model predictions are inconsistant so not recommended for use!{next_line}"
            f"{market_comments}.{next_line}"
        )
        send_email = 0
        
    return comment, send_email