import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from .data_processing import scenario_reverse, scenario_continue, model_test
from feature_engine.discretisation import EqualFrequencyDiscretiser


def load_file(file_path):
    #data management-loads the filepath to get the model
    return joblib.load(filename=file_path)


def predict_profits(X_live, model_feature, model_pipeline, model_label_map, y_actual):
    """
        This function put all the model components to generate the results.
        The results are formatted.

    Args:
        X_live (dataframe): dataframe contains model attributes and is one row of data
        model_feature (_type_): this is referenced to the plk file containing the key features
        model_pipeline (_type_): this conatins info about the pipline used to trained the model
        model_label_map (_type_): this is the categories used to split the y(dependent) variable

    Returns:
        _type_: return the formatted results
    """
   
    # from live data, subset features related to this pipeline
    X_live_subset = X_live.filter(model_feature)
    
    # predict the probability
    model_prediction_proba = model_pipeline.predict_proba(X_live_subset)
    
    #gets the maximum probability prediction categorical label
    model_prediction_labels = model_prediction_proba.argmax(axis=1)

    # Create a binary array based on the categorical value where value < 4 is a sell which gives True (1)
    # Convert both actual and prediction into binary numbers.
    binary_y_actual = (y_actual < 4).astype(int)
    binary_prediction= (model_prediction_labels < 4).astype(int)
    
    # Calculate the accuracy between the prediction and actual
    accuracy = calculate_accuracy(binary_prediction, binary_y_actual)

    print("Accuracy", accuracy)
    
    result_dict = {}
    #First loop goes through the probability profit/loss categories label, hard code range!
    #Second loop goes through the array containing the probability dictionary
    result_dict.update({
        "Predicted Profit": [model_label_map[model_prediction_labels[j]]
                        for j in range(model_prediction_labels.shape[0])
                    ]
        }
        )
    
    for i in range(len(model_label_map)):
        
                result_dict.update({
                f"{model_label_map[i]}": [
                    round(model_prediction_proba[j, i] * 100, 2) 
                    for j in range(model_prediction_proba.shape[0])
                ]
            }
            )
                
    
    print("prediction>>>>>>>", model_prediction_labels[0])
    print(model_label_map[1])
    return (result_dict)


def model_run(X_live):
    
    """
    This function gather all relevant model files needed to run the model. 
    Another function is called to generate and format the results.
    
    Args:
        X_live (dataframe): dataframe contains all attributes and can be more than one row.
                            The predict_profits() function will filter the relevant features.

    Returns:
        _type_: returns model results.
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
    
    
    # results = predict_profits(X_live, profit_features,
    #                             profit_pip, profit_labels_map)
   
  

    # Discretize the target variable
    disc = EqualFrequencyDiscretiser(q=6, variables=['pl_close_4_hr'])
    X_live_discretized = disc.fit_transform(X_live)
    y_actual = X_live_discretized['pl_close_4_hr']

    # Assuming predict_profits returns predictions
    results = predict_profits(X_live_discretized, profit_features, profit_pip, profit_labels_map, y_actual)



   
    return results

def standard_analysis():
    
    """
    This is a function to generate some standard analysis to show on the webpage.
    Pre-defined scenarios are inputted into the model.
    This calls on the model_run function which pulls all relevant inputs to generate results.

    Returns:
        _type_: returns results from model for the different scenarios
    """
    
    X_live_reverse = scenario_reverse()
    X_live_continue = scenario_continue()
    X_live_historical = model_test()
    
    pred_reverse = model_run( X_live_reverse )
    pred_continue = model_run( X_live_continue )
    print("historical input testing starts")
    pred_historical = model_run(X_live_historical)
    # pred_reverse, acc_rev = model_run( X_live_reverse )
    # pred_continue, acc_cont = model_run( X_live_continue )
    
    # print("accuracy reverse", acc_rev)
    # print("accuracy continue", acc_cont)
    
    return pred_reverse, pred_continue, X_live_reverse


def calculate_accuracy(predictions, y_actual):
    """
    Calculate the accuracy of the model.

    Args:
        predictions (array-like): Model predictions.
        y_actual (Series or array-like): Actual outcomes.

    Returns:
        accuracy (float): Accuracy of the model.
    """
    # You may use appropriate metrics based on your problem (e.g., accuracy_score)
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_actual, predictions)
    return accuracy