def comment_model_results(model_results_dict, model_results_label):
    array = model_results_dict[model_results_label][2]['item']['Potential Trade']
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


class ModelComparer:
    
    """
    This class compares results from three different models and provides comments based on consistency.
    
    Attributes:
        model_dict1 (dict): Results from the short term model.
        model_dict2 (dict): Results from the second model, typically for long-term.
        model_dict3 (dict): Results from the third model, typically for long-term.
        arr_index (int): An index in the array to extract model results for comparison.
        newline_syntax (int): 1 for '\n', otherwise <br>.
        comment (str): Analysis comments based on comparison results.
        send_email (bool): Indicates whether an email should be sent based on the comparison results.
    """
    
    def __init__(self, model_dict1, model_dict2, model_dict3, arr_index, newline_syntax):
        
        """
        Initializes the ModelComparer instance.

        Args:
            model_dict1 (dict): Results from the first model.
            model_dict2 (dict): Results from the second model.
            model_dict3 (dict): Results from the third model.
            arr_index (int): An index in the array to extract model results for comparison.
            newline_syntax (int): 1 for '\n', otherwise <br>.
        """
        
        self.model_dict1 = model_dict1
        self.model_dict2 = model_dict2
        self.model_dict3 = model_dict3
        self.arr_index = arr_index
        self.newline_syntax = newline_syntax
        self.comment = ""
        self.send_email = False
        self.trade_position = ""
        self.trade_target = 0
        self.bb_target = {}
        
        self.compare_versions()  # Call the comparison method during initialization

    def compare_versions(self):
        """
        Compares the results from three different models and sets the comment and send_email attributes accordingly.
         - current_trade reports the buy or sell call
         - pip_size reports the size of the movement
        """
        
        key_label1 = list(self.model_dict1.keys())[0]
        array1 = self.model_dict1[key_label1][2]['item']['Potential Trade']
        current_trade1 = array1[self.arr_index].split()[0]
        pip_size1 = abs(int(array1[self.arr_index].split()[2]))

        key_label2 = list(self.model_dict2.keys())[0]
        array2 = self.model_dict2[key_label2][2]['item']['Potential Trade']
        current_trade2 = array2[self.arr_index].split()[0]
        pip_size2 = abs(int(array2[self.arr_index].split()[2]))

        key_label3 = list(self.model_dict3.keys())[0]
        array3 = self.model_dict3[key_label3][2]['item']['Potential Trade']
        current_trade3 = array3[self.arr_index].split()[0]
        pip_size3 = abs(int(array3[self.arr_index].split()[2]))

        if self.newline_syntax:
            next_line = "\n"
        else:
            next_line = "<br>"

        if current_trade1 == current_trade2 == current_trade3:
            if pip_size2 >= 20:
                self.comment = f"All model versions predict a STRONG {current_trade1}{next_line}"
                self.send_email = True
                self.trade_position = current_trade1
                self.trade_target = pip_size1
            elif pip_size1 <= 10 or pip_size2 <= 10:
                self.comment = f"All model versions predict a weak {current_trade1}{next_line}"
                self.send_email = False
                self.trade_position = current_trade1
                self.trade_target = pip_size1
        elif current_trade1 == current_trade2 and current_trade3 != current_trade1:
            if pip_size3 >= 5:
                #shows a contradiction between long and short term position, but we take the short position
                self.comment = f"A {current_trade3} retracement or reverse point here!"
                self.send_email = False
                self.trade_position = current_trade3
                self.trade_target = pip_size3
            elif pip_size3 <= 5:
                self.comment = f"A weak {current_trade3} retracement or reverse point here!."
                self.send_email = False
                self.trade_position = current_trade3
                self.trade_target = pip_size3
        elif current_trade1 == current_trade3:
            if pip_size1 <= 5:
                self.comment = f"4hr model(v4) and 1hr model predict a weak {current_trade1}.{next_line}"
                self.send_email = True
                self.trade_position = current_trade1
                self.trade_target = pip_size1
            elif pip_size1 >= 5:
                self.comment = f"4hr model(v4) and 1hr model predict a strong {current_trade1}.{next_line}"
                self.send_email = True
                self.trade_position = current_trade1
                self.trade_target = pip_size1
            else:
                self.comment = f"WARNING: Not a good time to trade!{next_line}"
                self.send_email = False
                self.trade_position = current_trade3
                self.trade_target = pip_size3
        else:
            self.comment = f"WARNING: Not a good time to trade!{next_line}"
            self.send_email = False
            self.trade_position = current_trade3
            self.trade_target = pip_size3
            
        self.bb_entry()

    def bb_entry(self):
        key_label1 = list(self.model_dict1.keys())[0]
        upper_bb = self.model_dict1[key_label1][4]['bb4_results']['upper_bb4']
        lower_bb = self.model_dict1[key_label1][4]['bb4_results']['lower_bb4']
        
      
        self.bb_target = {"lower_bb":lower_bb[self.arr_index], "upper_bb":upper_bb[self.arr_index]}
     
            

def compare_version_results2(model_dict1, model_dict2, model_dict3, arr_index, newline_syntax):
    """
    The function provides comments based on consistency of models output. 

    Args:
        model_dict1 (dictionary): results from long term model
        model_dict2 (dictionary): results from long term model
        model_dict3 (dictionary): results from short term model
        prices_df (dataframe): a list of prices in dataframe
        arr_index (integer): an index in the array to extract model results for comparison - refer to the json file
        newline_syntax (integer): 1 for '\n' otherwise <br> for website usage

    Returns:
        string: analysis comments
    """
    
    key_label1 = list(model_dict1.keys())[0]
    array1 = model_dict1[key_label1][2]['item']['Potential Trade']
    current_trade1 = array1[arr_index].split()[0]
    pip_size1 = abs(int(array1[arr_index].split()[2]))
    
    key_label2 = list(model_dict2.keys())[0]
    array2 = model_dict2[key_label2][2]['item']['Potential Trade']
    current_trade2 = array2[arr_index].split()[0]
    pip_size2 = abs(int(array2[arr_index].split()[2]))
    
    key_label3 = list(model_dict3.keys())[0]
    array3 = model_dict3[key_label3][2]['item']['Potential Trade']
    current_trade3 = array3[arr_index].split()[0]
    pip_size3 = abs(int(array3[arr_index].split()[2]))
    
        
    if newline_syntax:
        next_line = "\n"
    else:
        next_line = "<br>"
    
    comment = ""
    send_email=False
    if current_trade1 == current_trade2 == current_trade3 :
        if pip_size2 >= 20:
            comment = (
                f"All model versions predict a STRONG {current_trade1}{next_line}"
            )
            send_email=True
        elif pip_size1 <= 10 or pip_size2 <=10:
            comment = (
                f"All model versions predict a weak {current_trade1}{next_line}"
            )
            send_email=False
                
    elif current_trade1 == current_trade2:
        if pip_size2 >= 20:
            comment = (
                f"Both 4hr models predict a STRONG {current_trade1}.{next_line}"
            )
            send_email=True
        elif pip_size1 <= 10:
            comment = (
                f"Both 4hr models predict a weak {current_trade1}.{next_line}"
            )
            send_email=False
    
    elif current_trade2 == current_trade3:
        
        if pip_size3 >= 5:
            comment = (
                f"4hr model and 1hr model predict a strong {current_trade2}.{next_line}"
            )
            send_email=True
        elif pip_size1 >= 10:
            comment = (
                f"4hr LAGGED model has a strong prediction for {current_trade1}.{next_line}"
            )
            send_email=False
        else:
            comment = (
            f"WARNING: Not a good time to trade!{next_line}"
            )
            send_email=False
             
    else:
        comment = (
            f"WARNING: Not a good time to trade!{next_line}"
        )
        send_email=False
        
    return comment, send_email


def compare_version_results(pred_collection, arr_index, newline_syntax):
    """
    The function provides comments based on consistency of models output. 

    Args:
        pred_collection (dictionary) : dictonary containing the predictions for different models
        prices_df (dataframe): a list of prices in dataframe
        arr_index (integer): an index in the array to extract model results for comparison - refer to the json file
        newline_syntax (integer): 1 for '\n' otherwise <br> for website usage

    Returns:
        string: analysis comments
    """
    
    model_rev1 = pred_collection["pred_reverse_v4"]
    model_rev2 = pred_collection["pred_reverse_v5"]
    model_rev3 = pred_collection["pred_reverse_1h_v5"]
       
    rev_label1 = list(model_rev1.keys())[0]
    rev_array1 = model_rev1[rev_label1][2]['item']['Potential Trade']
    rev_10pip_trade1 = rev_array1[arr_index].split()[0]
    rev_pip_size1 = abs(int(rev_array1[arr_index].split()[2]))
    
    rev_label2 = list(model_rev2.keys())[0]
    rev_array2 = model_rev2[rev_label2][2]['item']['Potential Trade']
    rev_10pip_trade2 = rev_array2[arr_index].split()[0]
    rev_pip_size2 = abs(int(rev_array2[arr_index].split()[2]))
    
    rev_label3 = list(model_rev3.keys())[0]
    rev_array3 = model_rev3[rev_label3][2]['item']['Potential Trade']
    rev_10pip_trade3 = rev_array3[arr_index].split()[0]
    rev_pip_size3 = abs(int(rev_array3[arr_index].split()[2]))
    
    model_cont1 = pred_collection["pred_continue_v4"]
    model_cont2 = pred_collection["pred_continue_v5"]
    model_cont3 = pred_collection["pred_continue_1h_v5"]
       
    cont_label1 = list(model_cont1.keys())[0]
    cont_array1 = model_cont1[cont_label1][2]['item']['Potential Trade']
    cont_10pip_trade1 = cont_array1[arr_index].split()[0]
    cont_pip_size1 = abs(int(cont_array1[arr_index].split()[2]))
    
    cont_label2 = list(model_cont2.keys())[0]
    cont_array2 = model_cont2[cont_label2][2]['item']['Potential Trade']
    cont_10pip_trade2 = cont_array2[arr_index].split()[0]
    cont_pip_size2 = abs(int(cont_array2[arr_index].split()[2]))
    
    cont_label3 = list(model_cont3.keys())[0]
    cont_array3 = model_cont3[cont_label3][2]['item']['Potential Trade']
    cont_10pip_trade3 = cont_array3[arr_index].split()[0]
    cont_pip_size3 = abs(int(cont_array3[arr_index].split()[2]))
    
        
    if newline_syntax:
        next_line = "\n"
    else:
        next_line = "<br>"
    
    comment = ""
    send_email=False
    if (rev_10pip_trade1 == rev_10pip_trade2 == rev_10pip_trade3) & (cont_10pip_trade1 == cont_10pip_trade2 == cont_10pip_trade3) :
        if rev_pip_size2 >= 5:
            comment = (
                f"For NEXT candle stick, all model versions predict a STRONG {rev_10pip_trade1}{next_line}"
            )
            send_email=True
                       
    elif (rev_10pip_trade1 == rev_10pip_trade2) & (cont_10pip_trade1 == cont_10pip_trade2) :
        if rev_pip_size2 >= 5:
            comment = (
                f"For Next candle stick, Both 4hr models predict a STRONG {rev_10pip_trade1}.{next_line}"
            )
            send_email=False
             
    else:
        comment = (
            f"WARNING: Not a good time to trade!{next_line}"
        )
        send_email=False
        
    return comment, send_email


def general_ticker_results(prices_df, newline_syntax):
    """
    The function provides general ticker information. 

    Args:
        prices_df (dataframe): a list of four prices with latest price at top and previous price at bottom
        newline_syntax (integer): 1 for '\n' otherwise <br> 

    Returns:
        string: analysis comments
    """
    
    date = prices_df.iloc[3]['date']
    open_price_1hr = round(prices_df.iloc[3]['open'], 2)
    open_price_4hr = round(prices_df.iloc[0]['open'], 2)
    close_price_1hr = round(prices_df.iloc[3]['close'], 2)
    entry_price_avg_1hr_4hr = round((prices_df.iloc[3]['open'] + prices_df.iloc[0]['open']) / 2, 2)
    
    
    if newline_syntax:
        next_line = "\n"
    else:
        next_line = "<br>"
    
    ticker_info = (
            f"Market Information:{next_line}"
            f"Date: {date}{next_line}"
            f"The current open price is {open_price_1hr}.{next_line}"
            f"The current close price is {close_price_1hr}.{next_line}"
            f"The price 4 hours ago is {open_price_4hr}.{next_line}"
            f"The average entry price 4 hours ago is {entry_price_avg_1hr_4hr}"
    )
    
    return ticker_info