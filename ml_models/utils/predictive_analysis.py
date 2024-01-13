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


def format_model_results(model_prediction_proba, model_prediction, model_label_map):
    """
        This function format the results form the model run.
        The results are formatted as dictionary so the html can report the results dynamically
        and not rely on harded coded categories.
        TODO: May explore other ways of storing the results as the dictionary is difficult to interpret how the results are saved.

    Args:
        model_prediction_proba (array): array contains arrays of probabilities to each categories predicted.
        model_prediction (array): array containing the prediction with max probability
        model_label_map (list): this is the categories used to split the trade variable (ie. y dependent)

    Returns:
        dictionary: return the formatted results
    """
   
   
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
    for j in range(model_prediction.shape[0]):
        direction = "Sell target: " if model_prediction[j] < 3 else "Buy target: "
        profit_label = model_label_map[model_prediction[j]]
        result_dict["Potential Trade"].append(
            f"{direction} {profit_label}"
        )
    
    return (result_dict)


def model_run(X_live):
    
    """
    This function calls on model pipline and generate the results as dataframe. 
    Another function is called to format the results into a dictionary format.
    
    Args:
        X_live (dataframe): dataframe contains live data of all attributes and can be more than one row.

    Returns:
        dictionary: returns model results, all predictions with probability, model predictions, model category label, 
        and live data used in the model
    """
    
    #get script directory
    script_directory = Path(__file__).resolve().parent

    # Move one level up
    parent_directory = script_directory.parent

    # Set the current working directory to the parent directory
    os.chdir(parent_directory)

    version = 'v4-new_py'
    model_pipeline = load_file(
        f"trained_models/USDJPY/pl_predictions/{version}/clf_pipeline.pkl")
    model_labels_map = load_file(
        f"trained_models/USDJPY/pl_predictions/{version}/label_map.pkl")
    model_features = (pd.read_csv(f"trained_models/USDJPY/pl_predictions/{version}/X_train.csv")
                       .columns
                       .to_list()
                       )
    
    # Discretize the target variable (ie. y dependent) 
    disc = EqualFrequencyDiscretiser(q=6, variables=['pl_close_4_hr'])
    X_live_discretized = disc.fit_transform(X_live)
    print("formated live data>>>>>>>>", (X_live_discretized))
    
    # extract the relevant subset features related to this pipeline
    X_live_subset = X_live_discretized.filter(model_features)
    
    # predict the probability for each of the cateogries
    model_prediction_proba = model_pipeline.predict_proba(X_live_subset)
    # print("probability>>>>>",model_prediction_proba)
    #prediction of model by getting the category with the maximum probability
    model_prediction = model_prediction_proba.argmax(axis=1)
    # print("Here are the model predictions >>>>>",model_prediction)
    
    # format the results of the model
    results = format_model_results(model_prediction_proba, model_prediction, model_labels_map)



    return results, model_prediction_proba, model_prediction, model_labels_map, X_live_discretized


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
    
    pred_reverse, _, _, _, _ = model_run(X_live_reverse)
    pred_continue, _, _, _, _ = model_run(X_live_continue)
    pred_historical, _, _, _, _ = model_run(X_live_historical)
     
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
    _ , model_prediction_proba, model_prediction, model_labels_map, X_live_discretized = model_run(X_live_historical)
   
    #combined the live data and prediction dataframes.
    df1 = pd.DataFrame(model_prediction_proba, columns=model_labels_map)
    df1 = df1.reset_index(drop=True)
    df2 = X_live_discretized.reset_index(drop=True)
    model_results = pd.concat([df1, df2 ], axis=1)
    model_results['prediction'] = model_prediction
    # print("model dataframe >>>>>", combined_df)
    # combined_df.to_csv(r"C:\Users\sunny\Desktop\Development\model_assessment_data.csv", index=False)
    
    # Create a binary array based on the categorical value where value < 3 is a sell (true is returned)
    # Convert both actual and prediction into binary numbers.
    # y_actual = X_live_discretized['pl_close_4_hr']
    # binary_y_actual = (y_actual < 3).astype(int)
    # binary_prediction= (model_prediction_labels < 3).astype(int)
    
    # Calculate the accuracy between the prediction and actual
    # accuracy = calculate_accuracy(binary_prediction, binary_y_actual)
    

    # print("Accuracy", accuracy)
    
    return model_results

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