import os
import sys
import joblib
import numpy as np
import pandas as pd
import datetime
from django.conf import settings
from tickers.models import Ticker
from prices.models import Price
from ml_models.utils.analysis_comments import compare_version_results, general_ticker_results, ModelComparer
from pathlib import Path
from feature_engine.discretisation import EqualFrequencyDiscretiser
from .access_results import write_to_json, read_prediction_from_json, write_to_csv
from django.shortcuts import get_object_or_404
from ml_models.utils.bespoke_model import v4Processing
from ml_models.utils.price_processing import StandardPriceProcessing
from ml_models.utils.reverse_model import reverse_model_run



def load_file(file_path):
    #data management-loads the filepath to get the model
    return joblib.load(filename=file_path)


def format_model_results(
        pred_name, heading_labels,
        model_prediction_proba,
        model_prediction, 
        model_labels_map,
        model_input_data):
    """
        This function format the results form the model run.
        The results are formatted as dictionary so the html can report the results dynamically
        and not rely on hard coded categories.

    Args:
        pred_name (string): the name of the scenario predction 
        heading_labels (array): an array of labels for headers in the html table
        model_prediction_proba (array): array contains arrays of probabilities to each categories predicted.
        model_prediction (array): array containing the prediction with the max probability
        model_label_map (list): this is the categories used to split the trade variable (ie. y dependent)

    Returns:
        dictionary: return results grouped as dictionary
    """
   
   
    #First loop goes through the probability profit/loss categories label
    main_results = {} 
    for i in range(len(model_labels_map)):
        
                main_results.update({
                f"{model_labels_map[i]}": [
                    round(model_prediction_proba[j, i] * 100, 2) 
                    for j in range(model_prediction_proba.shape[0])
                ]
            }
            )
    
    #Second loop goes through the array containing the probability dictionary
    potential_trade = []
    trade_type = []
    trade_target = []
    upper_bb4 =[]
    lower_bb4 =[]

    for row in model_input_data['upper_bb20_4']:
        upper_bb4.append(row)
        
    for row in model_input_data['lower_bb20_4']:
        lower_bb4.append(row)
 
    for prediction in model_prediction:
        direction = "Sell" if prediction < 3 else "Buy"
        profit_label = model_labels_map[prediction]
        potential_trade.append(f"{direction} target: {profit_label}")
        trade_type.append(direction)
        trade_target.append(profit_label)

    main_results["Potential Trade"] = potential_trade
    extra_results = {
        "trade_type": trade_type,
        "trade_target": trade_target
    }
    bb4_results = {
        "upper_bb4": upper_bb4,
        "lower_bb4": lower_bb4
    }

    # Adding the current timestamp in the format: day-month-year hour:minute:second
    current_timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
    group_results = {
        pred_name: [
            {"date": current_timestamp},
            {"heading": heading_labels},
            {"item": main_results},
            {"extra": extra_results},
            {"bb4_results": bb4_results},
        ]
    }
    
    return group_results


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
        
        # Rename the new column to indicate it's discretized
        original_column_name = y_test_headers[0]
        discretized_column_name = f"{original_column_name}_discretized"
        X_live_discretized.rename(columns={original_column_name: discretized_column_name}, inplace=True)
        
        # Keep the original column as well
        X_live_discretized[original_column_name] = X_live[original_column_name]
        
        # X_live_discretized.to_csv(r"C:\Users\sunny\Desktop\Development\discretized_data.csv", index=False)
    
    except:
        X_live_discretized = X_live
    
    X_live_discretized.reset_index(drop=True, inplace=True)
    
    # extract the relevant subset features related to this pipeline
    X_live_subset = X_live_discretized.filter(model_features)
    # predict the probability for each of the cateogries
    model_prediction_proba = model_pipeline.predict_proba(X_live_subset)
    
    #prediction of model by getting the category with the maximum probability
    model_prediction = model_prediction_proba.argmax(axis=1)
    
    # # format the results of the model
    # results, extra_results = format_model_results(model_prediction_proba, model_prediction, model_labels_map)
    
    return {
            'model_prediction_proba': model_prediction_proba,
            'model_prediction': model_prediction,
            'model_labels_map': model_labels_map,
            'X_live_discretized': X_live_discretized
        }


def standard_analysis(ticker, model_version, sensitivity_adjustment=0.1):
    
    """
    This is a function to generate some standard analysis to show on the webpage.
    Pre-defined scenarios are inputted into the model.
    This calls on the model_run function which pulls all relevant inputs to generate results.
    Json file will be produced saving the results.
    Returns:
        dictionary: returns results from model for the different scenarios
    """
    
    # Dynamically get the module path involves defining the parent directory
    # parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # # Add the parent directory to the Python path
    # sys.path.append(parent_directory)
    # module_name = f'trained_models.{ticker}.pl_predictions.{model_version}.data_processing'

    # try:
    #     dp = importlib.import_module(module_name)
    # except ImportError:
    #     print(f"Error importing data_processing module for model_version: {model_version}")
    #     dp = None
    
    if model_version == "v4":
        dp = v4Processing(ticker=ticker)
    else:
        dp = StandardPriceProcessing(ticker=ticker)
    
    X_live_reverse = dp.scenario_reverse()
    X_live_continue = dp.scenario_continue()
    X_live_historical = dp.historical_record(4)
    X_live_variability = dp.prediction_variability(sensitivity_adjustment)
    
    
    pred_reverse_results = model_run(ticker, X_live_reverse, model_version)
    pred_continue_results = model_run(ticker, X_live_continue, model_version)
    pred_historical_results = model_run(ticker, X_live_historical, model_version)
    pred_variability_results = model_run(ticker, X_live_variability, model_version)
      

    pred_reverse = format_model_results(
                        pred_name="pred_reverse", 
                        heading_labels=["Current", "10pips Reversed", "20pips Reversed"], 
                        model_prediction_proba=pred_reverse_results['model_prediction_proba'],
                        model_prediction=pred_reverse_results['model_prediction'],
                        model_labels_map=pred_reverse_results['model_labels_map'],
                        model_input_data=pred_reverse_results['X_live_discretized']
                    )
    pred_continue = format_model_results(
                        pred_name="pred_continue", 
                        heading_labels=["Current", "10pips Continue", "20pips Continue"], 
                        model_prediction_proba=pred_continue_results['model_prediction_proba'],
                        model_prediction=pred_continue_results['model_prediction'],
                        model_labels_map=pred_continue_results['model_labels_map'],
                        model_input_data=pred_continue_results['X_live_discretized']
                    )
    pred_historical = format_model_results(
                        pred_name="pred_historical", 
                        heading_labels=["3hr ago", "2hr ago", "1hr ago", "Current"], 
                        model_prediction_proba=pred_historical_results['model_prediction_proba'],
                        model_prediction=pred_historical_results['model_prediction'],
                        model_labels_map=pred_historical_results['model_labels_map'],
                        model_input_data=pred_historical_results['X_live_discretized']
                    )
    pred_variability = format_model_results(
                        pred_name="pred_variability", 
                        heading_labels=["10pips", "-10pips"], 
                        model_prediction_proba=pred_variability_results['model_prediction_proba'],
                        model_prediction=pred_variability_results['model_prediction'],
                        model_labels_map=pred_variability_results['model_labels_map'],
                        model_input_data=pred_variability_results['X_live_discretized']
                    )
    
    
    write_to_json(pred_reverse, ticker, f"{ticker}_pred_reverse_{model_version}.json")
    write_to_json(pred_continue, ticker, f"{ticker}_pred_continue_{model_version}.json")
    write_to_json(pred_historical, ticker, f"{ticker}_pred_historical_{model_version}.json")
    write_to_json(pred_variability, ticker, f"{ticker}_pred_variability_{model_version}.json")
     
    return pred_reverse, pred_continue, pred_historical, pred_variability


def trade_forecast_assessment(ticker, model_version):
    """
    This function is to produce the model results use for assessment or exporting.

    Args:
        model_version (string): the model version to produce the results for assessment.
        
    Returns:
        dataframe: a dataframe containing the results
    """
    
    if model_version == "v4" or model_version == "v1_reverse":
        dp = v4Processing(ticker=ticker)
    else:
        dp = StandardPriceProcessing(ticker=ticker)
    
    
    X_live_historical = dp.historical_record(120)
    
    if model_version == "v1_reverse":
        pred_historical = reverse_model_run(ticker, X_live_historical, model_version)
    else:
        pred_historical = model_run(ticker, X_live_historical, model_version)
    
    model_prediction_proba = pred_historical['model_prediction_proba']
    model_labels_map = pred_historical['model_labels_map']
    model_prediction = pred_historical['model_prediction']
    X_live_discretized = pred_historical['X_live_discretized']
   
    #combined the live data and prediction dataframes.
    df1 = pd.DataFrame(model_prediction_proba, columns=model_labels_map)
    df1 = df1.reset_index(drop=True)
    df2 = X_live_discretized.reset_index(drop=True)
    model_results = pd.concat([df1, df2 ], axis=1)
    model_results['prediction'] = model_prediction
 
    # combined_df.to_csv(r"C:\Users\sunny\Desktop\Development\model_assessment_data.csv", index=False)
    
    #Getting the header of the y variable
    version = model_version
    y_test_headers = (pd.read_csv(f"trained_models/{ticker}/pl_predictions/{version}/y_test.csv")
                       .columns
                       .to_list()
                       )
    

    # Using engine feature package to classify the actual numbers. 
    # Note that the EqualFrequencyDiscretiser only takes in a string for the variables, not a list;
    # hence, we use an index. Another way is use bin.
    # TODO: the q is hardcoded as 6 which may require to be made dynamic counting the size o the label map.
    disc = EqualFrequencyDiscretiser(q=6, variables=[y_test_headers[0]])
    y_actual = disc.fit_transform(X_live_discretized[y_test_headers])
     
    # Calculate the accuracy between the prediction and actual
    profit_accuracy = calculate_accuracy(model_prediction, y_actual)

    # this examine the accuracy of the trade predictions converted to buy(1) or sell(0)
    y_actual_binary = np.where(y_actual > 2, 0, 1)
    model_prediction_binary = np.where(model_prediction > 2, 0, 1)
    trade_accuracy = calculate_accuracy(model_prediction_binary, y_actual_binary)

    
    return model_results, profit_accuracy, trade_accuracy


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


def run_model_predictions(model_ticker, sensitivity_adjustment=0.1):
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


def variability_analysis(model_ticker, sensitivity_adjustment):
    """
        Perform variability analysis for multiple model versions depending on the sensitivity adjustment.

    Args:
        model_ticker (str): The ticker symbol of the model.
        sensitivity_adjustment (float): The sensitivity adjustment value for prediction variability.

    Returns:
        string: Comments on how the three model compare and what trading direction to take.
    """
    
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_directory)
    
    model_versions = ["v4", "v5", "1h_v5"]
    pred_variability_results = {}  # Dictionary to store results
    
    for model_version in model_versions:

        if model_version == "v4":
            dp = v4Processing(ticker=model_ticker)
        else:
            dp = StandardPriceProcessing(ticker=model_ticker)
        
        X_live_variability = dp.prediction_variability(sensitivity_adjustment)
        pred_variability = model_run(model_ticker, X_live_variability, model_version)
        pred_variability_results[model_version]  = format_model_results(
                                                        pred_name="pred_variability", 
                                                        heading_labels=[sensitivity_adjustment, -sensitivity_adjustment], 
                                                        model_prediction_proba=pred_variability['model_prediction_proba'],
                                                        model_prediction=pred_variability['model_prediction'],
                                                        model_labels_map=pred_variability['model_labels_map'],
                                                        model_input_data=pred_variability['X_live_discretized']
                                                    )
        
    pred_historical_v4 = pred_variability_results[model_versions[0]]
    pred_historical_v5 = pred_variability_results[model_versions[1]]
    pred_historical_1h_v5 = pred_variability_results[model_versions[2]]
    
    model_comparer_pos = ModelComparer(pred_historical_v4, pred_historical_v5, pred_historical_1h_v5, 0, 1 )
    model_comparer_neg = ModelComparer(pred_historical_v4, pred_historical_v5, pred_historical_1h_v5, 1, 1 )
    version_comment_pos = model_comparer_pos.comment
    version_comment_neg = model_comparer_neg.comment
    
    return version_comment_pos, version_comment_neg