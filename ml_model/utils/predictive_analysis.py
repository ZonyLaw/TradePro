import os
import joblib
import pandas as pd
from pathlib import Path
from .data_processing import scenario_reverse, scenario_continue
import csv


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
    model_prediction_proba = model_pipeline.predict_proba(X_live_subset)

    #delete
    result_dict ={
        f"{model_label_map[i]}": round(model_prediction_proba[0, i] * 100,2)
        for i in range(6)
    }

    
    result_dict2 = {}
    for i in range(6):
            result_dict2.update({
                f"{model_label_map[i]}": [
                    round(model_prediction_proba[0, i] * 100,2),
                    round(model_prediction_proba[1, i] * 100,2),
                    round(model_prediction_proba[2, i] * 100,2),
                ]
            }
            )

    #delete
    result_df = pd.DataFrame(data= model_prediction_proba)
    new_label = {result_df.columns.values[0]:f'{model_label_map[0]}',
                 result_df.columns.values[1]:f'{model_label_map[1]}',
                 result_df.columns.values[2]:f'{model_label_map[2]}',
                 result_df.columns.values[3]:f'{model_label_map[3]}',
                 result_df.columns.values[4]:f'{model_label_map[4]}',
                 result_df.columns.values[5]:f'{model_label_map[5]}'}
    
    result_df.rename(columns = new_label, inplace = True)
    result_html = result_df.to_html()
    # print("html-----------",result_html)
    # print("dataframe>>>>>fun", result_df)
    
    print("predict profit model>>>>", model_prediction_proba[0,1])
    print("result_dict2.....", result_dict2)
    return (result_dict2)


def model_run():
    
    """
    This function handles the key input components and load the trained model. 
    The formatting and production of results are handled by another function.   

    Returns:
        _type_: returns calls on another function generating and formating of the results.
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
    
    X_live_reverse = scenario_reverse()
    X_live_continue = scenario_continue()
    
    pred_reverse = predict_profits(X_live_reverse, profit_features,
                                profit_pip, profit_labels_map)
   
    pred_continue = predict_profits(X_live_continue, profit_features,
                                profit_pip, profit_labels_map)
   
    return pred_reverse, pred_continue