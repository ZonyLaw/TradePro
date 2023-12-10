import os
import joblib
import pandas as pd
from pathlib import Path


def load_file(file_path):
    #data management-loads the filepath to get the model
    return joblib.load(filename=file_path)


def predict_profits(X_live, model_feature, model_pipeline, model_label_map):
    """
        This function takes model components and feed the data to generate the results.
        The results are formatted.

    Args:
        X_live (dataframe): dataframe contains model attributes and is one row of data
        model_feature (_type_): this is referenced to the plk file containing the key features
        model_pipeline (_type_): this conatins info about the pipline used to trained the model
        model_label_map (_type_): this is the categories used to split the y(dependent) variable

    Returns:
        _type_: return the formatted results
    """

    category_phrase = {
        '-25<': 'Sell with >25 pips target',
        '-20': 'Sell with 20 pips target',
        '-5': 'Sell with 5 pips target',
        '5': 'Buy with 5 pips target',
        '20': 'Buy with 20 pips target',
        '>25': 'Buy with >25 pips target'
    }
    
    # from live data, subset features related to this pipeline
    X_live_subset = X_live.filter(model_feature)
    
    # predict
    model_prediction = model_pipeline.predict(X_live_subset)
    model_prediction_proba = model_pipeline.predict_proba(X_live_subset)
    # st.write(model_prediction_proba)

    # create a logic to display the results
    proba = model_prediction_proba[0, model_prediction][0]*100
    category_labels = model_label_map[model_prediction[0]]

    # output the entire array    
    proba1 = model_prediction_proba[0, 0]*100
    category_labels1 = model_label_map[0]
    proba2 = model_prediction_proba[0, 1]*100
    category_labels2 = model_label_map[1]
    proba3 = model_prediction_proba[0, 2]*100
    category_labels3 = model_label_map[2]
    proba4 = model_prediction_proba[0, 3]*100
    category_labels4 = model_label_map[3]
    proba5 = model_prediction_proba[0, 4]*100
    category_labels5 = model_label_map[4]
    proba6 = model_prediction_proba[0, 5]*100
    category_labels6 = model_label_map[5]

   
    print( model_prediction_proba[0,0]*100)

    
    statement = (
        f"* If you are in this trade, there is a {proba.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels]} pips**.\n\n"
        
        f"-------------------------------All statistics-------------------------------\n\n"
        
        f"* If you are in this trade, there is a {proba1.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels1]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba2.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels2]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba3.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels3]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba4.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels4]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba5.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels5]} pips**.\n\n"
        f"* If you are in this trade, there is a {proba6.round(2)}% probability the profit can "
        f"reach **{category_phrase[category_labels6]} pips**.\n\n"

    )
    
    print(statement)
    return (statement)


def model_run():
    
    """
    This groups all the key components to run the trained model where another function handles the formatting of the results.  
    

    Returns:
        _type_: returns the results from the model
    """
    
    #get script directory
    script_directory = Path(__file__).resolve().parent

    # Move one level up
    parent_directory = script_directory.parent

    # Set the current working directory to the parent directory
    os.chdir(parent_directory)

    version = 'v4'
    profit_pip = load_file(
        f"trained_models/USDJPY/pl_predictions/{version}/clf_pipeline.pkl")
    profit_labels_map = load_file(
        f"trained_models/USDJPY/pl_predictions/{version}/label_map.pkl")
    profit_features = (pd.read_csv(f"trained_models/USDJPY/pl_predictions/{version}/X_train.csv")
                       .columns
                       .to_list()
                       )
    
    #creating some input to the model
    X_live = pd.DataFrame([], index=[0])
    X_live.loc[0, 'open_close_diff_1'] = -0.5
    X_live.loc[0, 'open_close_diff1_lag1'] = 0.1
    X_live.loc[0, 'close_ma50_1_diff_1'] = -0.5
    X_live.loc[0, 'bb_status_1'] = "upper_near"
    X_live.loc[0, 'up_bb20_low_bb20_diff_1'] = 1.5
    X_live.loc[0, 'trend_strength_1'] = 1
    X_live.loc[0, 'lagged_close_1'] = 144.80
    X_live.loc[0, 'hr'] = 8
    X_live.loc[0, 'up_bb20_low_bb20_diff_4'] = 6.0
    X_live.loc[0, 'ma50_4_ma100_4_diff_4'] = -0.5
    X_live.loc[0, 'ma20_4_ma50_4_diff_4'] = -0.5
    X_live.loc[0, 'close_ma100_4_diff_4'] = -0.4

    # Print the resulting DataFrame
    print(X_live)

    return predict_profits(X_live, profit_features,
                                profit_pip, profit_labels_map)