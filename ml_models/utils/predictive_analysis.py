import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from .data_processing import scenario_reverse, scenario_continue, historical_record
from feature_engine.discretisation import EqualFrequencyDiscretiser


def load_file(file_path):
    #data management-loads the filepath to get the model
    return joblib.load(filename=file_path)


def predict_trade(X_live, model_feature, model_pipeline, model_label_map):
    """
        This function put all the model components to generate the results.
        The results are formatted as dictionary so the html can report the results dynamically
        and not rely on harded coded categories.
        TODO: May explore other ways of storing the results as the dictionary is difficult to interpret how the results are saved.

    Args:
        X_live (dataframe): dataframe contains model attributes and is one row of data
        model_feature (sklearn.pipeline.Pipeline): this is referenced to the plk file containing the key features
        model_pipeline (sklearn.pipeline.Pipeline): this conatins info about the pipline used to trained the model
        model_label_map (list): this is the categories used to split the trade variable (ie. y dependent)

    Returns:
        dictionary: return the formatted results
    """
   
    # from live data, subset features related to this pipeline
    X_live_subset = X_live.filter(model_feature)
    
    # predict the probability
    model_prediction_proba = model_pipeline.predict_proba(X_live_subset)
    # print("probability",model_prediction_proba)
    #gets the maximum probability prediction categorical label
    model_prediction_labels = model_prediction_proba.argmax(axis=1)
    # print("here are the label",model_prediction_labels)
   
    #First loop goes through the probability profit/loss categories label
    #Second loop goes through the array containing the probability dictionary
    result_dict = {} 
    for i in range(len(model_label_map)):
        
                result_dict.update({
                f"{model_label_map[i]}": [
                    round(model_prediction_proba[j, i] * 100, 2) 
                    for j in range(model_prediction_proba.shape[0])
                ]
            }
            )
    
    result_dict["Potential Trade"]=[]
    for j in range(model_prediction_labels.shape[0]):
        direction = "Sell target: " if model_prediction_labels[j] < 3 else "Buy target: "
        profit_label = model_label_map[model_prediction_labels[j]]
        result_dict["Potential Trade"].append(
            f"{direction} {profit_label}"
        )
    
    return (result_dict)


def model_run(X_live):
    
    """
    This function gather all relevant model files needed to run the model. 
    Another function is called to generate and format the results.
    
    Args:
        X_live (dataframe): dataframe contains all attributes and can be more than one row.
                            The predict_trade() function will filter the relevant features.

    Returns:
        dictionary: returns model results.
    """
    
    #get script directory
    script_directory = Path(__file__).resolve().parent

    # Move one level up
    parent_directory = script_directory.parent

    # Set the current working directory to the parent directory
    os.chdir(parent_directory)

    version = 'v4-new_py'
    profit_pip = load_file(
        f"trained_models/USDJPY/pl_predictions/{version}/clf_pipeline.pkl")
    profit_labels_map = load_file(
        f"trained_models/USDJPY/pl_predictions/{version}/label_map.pkl")
    profit_features = (pd.read_csv(f"trained_models/USDJPY/pl_predictions/{version}/X_train.csv")
                       .columns
                       .to_list()
                       )
    
    # Discretize the target variable
    disc = EqualFrequencyDiscretiser(q=6, variables=['pl_close_4_hr'])
    X_live_discretized = disc.fit_transform(X_live)

    # Assuming predict_trade returns predictions
    results = predict_trade(X_live_discretized, profit_features, profit_pip, profit_labels_map)

    return results

def standard_analysis():
    
    """
    This is a function to generate some standard analysis to show on the webpage.
    Pre-defined scenarios are inputted into the model.
    This calls on the model_run function which pulls all relevant inputs to generate results.

    Returns:
        dictionary: returns results from model for the different scenarios
    """
    
    X_live_reverse = scenario_reverse()
    X_live_continue = scenario_continue()
    X_live_historical = historical_record(4)
    
    pred_reverse = model_run(X_live_reverse)
    pred_continue = model_run(X_live_continue)
    pred_historical = model_run(X_live_historical)
    test = trade_forecast_assessment
    print("looking at pre_historical>>>>>", test)  
     
    return pred_reverse, pred_continue, pred_historical


def trade_forecast_assessment():
    """
    This function is to assess the trade forecast accuracy and produce a csv file for further analysis.

    Args:
        y_actual (): _description_

    Returns:
        _type_: _description_
    """
    
    X_live_historical = historical_record(60)
    pred_historical = model_run(X_live_historical)
    
   
    # Create a binary array based on the categorical value where value < 3 is a sell (true is returned)
    # Convert both actual and prediction into binary numbers.
    # y_actual = X_live_discretized['pl_close_4_hr']
    # binary_y_actual = (y_actual < 3).astype(int)
    # binary_prediction= (model_prediction_labels < 3).astype(int)
    
    # Calculate the accuracy between the prediction and actual
    # accuracy = calculate_accuracy(binary_prediction, binary_y_actual)
    

    # print("Accuracy", accuracy)
    
    return pred_historical

def calculate_accuracy(predictions, y_actual):
    """
    Calculates the accuracy of the model.

    Args:
        predictions (array-like): Model predictions.
        y_actual (Series or array-like): Actual outcomes.

    Returns:
        (float): Accuracy of the model.
    """
    # You may use appropriate metrics based on your problem (e.g., accuracy_score)
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_actual, predictions)
    return accuracy