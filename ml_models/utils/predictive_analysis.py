import os
import sys
import joblib
import importlib.util
import pandas as pd
import datetime
from django.conf import settings
from tickers.models import Ticker
from prices.models import Price
from ml_models.utils.analysis_comments import compare_version_results, general_ticker_results
from pathlib import Path
from feature_engine.discretisation import EqualFrequencyDiscretiser
from .access_results import write_to_json, read_prediction_from_json, write_to_csv
from django.shortcuts import get_object_or_404

def load_file(file_path):
    #data management-loads the filepath to get the model
    return joblib.load(filename=file_path)


def transform_format(pred_name, heading_labels, original_data):
    """
    This is to add the name tag to the prediction results and some heading labels to the data set
    used for producing the table in html.

    Args:
        pred_name (string): the name of the scenario predction 
        heading_labels (array): an array of labels for headers in the html table
        original_data (dictionary): the model prediction results in dictionary format

    Returns:
        dictionary: the new dictionary with scenario predction name and header to the table.
    """
    
    # Adding the current timestamp in the format: day-month-year hour:minute:second
    current_timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
    transformed_data = {
        pred_name: [
            {"date": current_timestamp},
            {"heading": heading_labels},
            {"item": original_data}
        ]
    }
    
    return transformed_data


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


def model_run(ticker, X_live, model_version):
    
    """
    This function calls on model pipline and generate the results as dataframe. 
    Another function is called to format the results into a dictionary format.
    
    Args:
        ticker (string): the ticker to run the model
        X_live (dataframe): dataframe contains live data of all attributes and can be more than one row.
        model_version: the model version to run

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

    version = model_version
    model_pipeline = load_file(
        f"trained_models/{ticker}/pl_predictions/{version}/clf_pipeline.pkl")
    model_labels_map = load_file(
        f"trained_models/{ticker}/pl_predictions/{version}/label_map.pkl")
    model_features = (pd.read_csv(f"trained_models/{ticker}/pl_predictions/{version}/X_test.csv")
                       .columns
                       .to_list()
                       )
    
    # Discretize the target variable (ie. y dependent) 
    
    y_test_headers = (pd.read_csv(f"trained_models/{ticker}/pl_predictions/{version}/y_test.csv")
                       .columns
                       .to_list()
                       )
    try:
        disc = EqualFrequencyDiscretiser(q=6, variables=[y_test_headers[0]])
        X_live_discretized = disc.fit_transform(X_live)
        # print("formated live data>>>>>>>>", (X_live_discretized))
    except:
        X_live_discretized = X_live
    
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


def standard_analysis(ticker, model_version):
    
    """
    This is a function to generate some standard analysis to show on the webpage.
    Pre-defined scenarios are inputted into the model.
    This calls on the model_run function which pulls all relevant inputs to generate results.
    Json file will be produced saving the results.
    Returns:
        dictionary: returns results from model for the different scenarios
    """
    
    # Dynamically get the module path involves defining the parent directory
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Add the parent directory to the Python path
    sys.path.append(parent_directory)
    module_name = f'trained_models.{ticker}.pl_predictions.{model_version}.data_processing'

    try:
        dp = importlib.import_module(module_name)
    except ImportError:
        print(f"Error importing data_processing module for model_version: {model_version}")
        dp = None
    
    
    X_live_reverse = dp.scenario_reverse()
    X_live_continue = dp.scenario_continue()
    X_live_historical = dp.historical_record(4)
    X_live_variability = dp.prediction_variability(0.10)
    
    
    pred_reverse, _, _, _, _ = model_run(ticker, X_live_reverse, model_version)
    pred_continue, _, _, _, _ = model_run(ticker, X_live_continue, model_version)
    pred_historical, _, _, _, _ = model_run(ticker, X_live_historical, model_version)
    pred_variability, _, _, _, _ = model_run(ticker, X_live_variability, model_version)
    
    
    pred_reverse = transform_format(f"pred_reverse", ["Current", "20pips Reversed", "40pips Reversed"], pred_reverse)
    pred_continue = transform_format(f"pred_continue", ["Current", "20pips Continue", "40pips Continue"], pred_continue)
    pred_historical = transform_format(f"pred_historical", ["3hr ago", "2hr ago", "1hr ago", "Current"], pred_historical)
    pred_variability = transform_format(f"pred_variability", ["10pips", "-10pips"], pred_variability)
    
    
    write_to_json(pred_reverse, ticker, f"{ticker}_pred_reverse_{model_version}.json")
    write_to_json(pred_continue, ticker, f"{ticker}_pred_continue_{model_version}.json")
    write_to_json(pred_historical, ticker, f"{ticker}_pred_historical_{model_version}.json")
    write_to_json(pred_variability, ticker, f"{ticker}_pred_variability_{model_version}.json")
     
    return pred_reverse, pred_continue, pred_historical, pred_variability


def trade_forecast_assessment(model_version):
    """
    This function is to assess the trade forecast accuracy and produce a csv file for further analysis.

    Args:
        y_actual (): _description_

    Returns:
        _type_: _description_
    """
    
    
    # Dynamically get the module path involves defining the parent directory
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Add the parent directory to the Python path
    sys.path.append(parent_directory)
    module_name = f'trained_models.USDJPY.pl_predictions.{model_version}.data_processing'

    try:
        dp = importlib.import_module(module_name)
    except ImportError:
        print(f"Error importing data_processing module for model_version: {model_version}")
        dp = None
    
    
    X_live_historical = dp.historical_record(120)
    _ , model_prediction_proba, model_prediction, model_labels_map, X_live_discretized = model_run("USDJPY", X_live_historical, model_version)
   
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


def run_model_predictions(model_ticker):
    """
    This function set off model run for three different versions and compares results.
    It will return the comments and ticker info which can be sent out by email if enabled.
    
    TODO: Currently set as USDJPY, but for future update the ticker probably need to be dynamic.
    """
    
    pred_reverse_v4, _, _, _ = standard_analysis(model_ticker, "v4")
    pred_reverse_v5, _, _, _ = standard_analysis(model_ticker,"v5")
    pred_reverse_1h_v5, _, _, _ = standard_analysis(model_ticker, "1h_v5")

    ticker_instance = get_object_or_404(Ticker, symbol=model_ticker)
    prices = Price.objects.filter(ticker=ticker_instance)
    
    #sort prices table in ascending so latest price on the bottom
    #note that html likes to work with array if using indexing
    prices_df = pd.DataFrame(list(prices.values()))
    sorted_prices_df = prices_df.sort_values(by='date', ascending=True)
    last_four_prices_df = sorted_prices_df.tail(4)
        
    comparison_comment, send_email_enabled = compare_version_results(pred_reverse_v4, pred_reverse_v5, pred_reverse_1h_v5, 0, 1 )
    general_ticker_info = general_ticker_results(last_four_prices_df, 1)

    print("test>>>>>>>>", comparison_comment)
    return comparison_comment, general_ticker_info, send_email_enabled


def variability_analysis(model_ticker):
    """
    This function is carry out a sensitivity test on 15 and -15pips movements.

    Args:
        model_ticker (string): the ticker to run the variability test

    Returns:
        string: comments on the variability test for positive and negative movements.
    """
    
    pred_variability_v4 = read_prediction_from_json(model_ticker, f'USDJPY_pred_variability_v4.json')
    pred_variability_v5 = read_prediction_from_json(model_ticker, f'USDJPY_pred_variability_v5.json')
    pred_variability_1h_v5 = read_prediction_from_json(model_ticker, f'USDJPY_pred_variability_1h_v5.json')
    
    
    version_comment_pos, _ = compare_version_results(pred_variability_v4, pred_variability_v5, pred_variability_1h_v5, 0, 1 )
    version_comment_neg, _ = compare_version_results(pred_variability_v4, pred_variability_v5, pred_variability_1h_v5, 1, 1 )
    
    write_to_csv(version_comment_pos, version_comment_neg, "variability_results.csv")
    
    return version_comment_pos, version_comment_neg