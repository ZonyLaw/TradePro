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
from ml_models.utils.reverse_model import standard_analysis_reverse
from pathlib import Path
from feature_engine.discretisation import EqualFrequencyDiscretiser
from .access_results import write_to_json, write_to_mongo, read_prediction_from_json, write_to_csv
from django.shortcuts import get_object_or_404
from ml_models.utils.bespoke_model import v4Processing
from ml_models.utils.price_processing import StandardPriceProcessing
from ml_models.utils.reverse_model import reverse_model_run
from ml_models.utils.trade import trade_direction



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
                f"prob{model_labels_map[i]}": [
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
    upper_bb1 =[]
    lower_bb1 =[]
    flatness_up_bb1_5=[]

   
    for i in range(len(model_input_data['upper_bb20_4'])):
        upper_bb4.append(model_input_data['upper_bb20_4'][i])
        lower_bb4.append(model_input_data['lower_bb20_4'][i])
        upper_bb1.append(model_input_data['upper_bb20_1'][i])
        lower_bb1.append(model_input_data['lower_bb20_1'][i])
        flatness_up_bb1_5.append(model_input_data['up_bb20_1_flat_5'][i])
 
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
    
    bb1_results = {
        "upper_bb1": upper_bb1,
        "lower_bb1": lower_bb1
    }
    
    bb4_results = {
        "upper_bb4": upper_bb4,
        "lower_bb4": lower_bb4
    }
    
    flatness_indicator = {
        "flatness": flatness_up_bb1_5,
    }

    # Adding the current timestamp in the format: day-month-year hour:minute:second
    current_timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
    group_results = {
        pred_name: 
            {"date": current_timestamp,
            "trade_headers": heading_labels,
            "trade_results": main_results,
            "trade_breakdown": extra_results,
            "bb1_results": bb1_results,
            "bb4_results": bb4_results,
            "flatness_indicator": flatness_indicator
             }
        
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
    
    if model_version == "v4" or model_version == "v5":
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
    
    #This can be removed in the future but save it in case MongoDB fails
    # print("JSON file saving starts here: >>>>>")
    # write_to_json(pred_reverse, ticker, f"{ticker}_pred_reverse_{model_version}.json")
    # write_to_json(pred_continue, ticker, f"{ticker}_pred_continue_{model_version}.json")
    # write_to_json(pred_historical, ticker, f"{ticker}_pred_historical_{model_version}.json")
    # write_to_json(pred_variability, ticker, f"{ticker}_pred_variability_{model_version}.json")
    
    print("MongoDB saving starts here: >>>>>")
    write_to_mongo(f'{ticker}_pred_reverse_{model_version}', pred_reverse)
    write_to_mongo(f'{ticker}_pred_continue_{model_version}', pred_continue)
    write_to_mongo(f'{ticker}_pred_historical_{model_version}', pred_historical)
    write_to_mongo(f'{ticker}_pred_variability_{model_version}', pred_variability)
    

    
     
    return pred_reverse, pred_continue, pred_historical, pred_variability


def trade_forecast_assessment(ticker, model_version):
    """
    This function is to produce the model results use for assessment or exporting.

    Args:
        model_version (string): the model version to produce the results for assessment.
        
    Returns:
        dataframe: a dataframe containing the results
    """
    
    if model_version == "v4" or model_version == "v1_reverse" or model_version == "v5":
        dp = v4Processing(ticker=ticker)
    else:
        dp = StandardPriceProcessing(ticker=ticker)
    
    
    X_live_historical = dp.historical_record(10)
  
    
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
    y_actual_binary = np.where(y_actual > 2, 1, 0)
    model_prediction_binary = np.where(model_prediction > 2, 1, 0)
    trade_accuracy = calculate_accuracy(model_prediction_binary, y_actual_binary)

    buy_sell_split = calc_matches(y_actual_binary.flatten(), model_prediction_binary)
 
    return model_results, profit_accuracy, trade_accuracy, buy_sell_split


def calc_matches(actual_array, model_array):
    """
    Takes in two arrays containing the actual and model results. These are compared to calculate
    the percentages of matches and mismatches. This information is intended to give an indication
    of the sentiment.

    Parameters:
    actual_array (array): Actual results (binary values).
    model_array (array): Model predictions (binary values).

    Returns:
    dict: A dictionary containing the percentage of sell matches, buy matches, and unmatched.
    """

    # Check if the lengths of the arrays match
    if len(actual_array) != len(model_array):
        raise ValueError("The lengths of actual_array and model_array must be the same.")

    # Create a DataFrame from the arrays
    df = pd.DataFrame({
        'actual': actual_array,
        'model_pred': model_array
    })

    # Compare the two columns to find matches
    sell_matches = (df['actual'] == df['model_pred']) & (df['model_pred'] == 0)
    buy_matches = (df['actual'] == df['model_pred']) & (df['model_pred'] == 1)

    # Count matches and calculate percentages
    total_elements = len(df)
    total_sell_matches = sell_matches.sum()
    total_buy_matches = buy_matches.sum()

    per_sell_matches = (total_sell_matches / total_elements) * 100 if total_elements > 0 else 0
    per_buy_matches = (total_buy_matches / total_elements) * 100 if total_elements > 0 else 0
    unmatched = 100 - per_sell_matches - per_buy_matches

    # Return the results in a dictionary
    return {
        "sell_matches": per_sell_matches,
        "buy_matches": per_buy_matches,
        "unmatched": unmatched
    }


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

from .access_results import write_to_mongo

def run_model_predictions(model_ticker, sensitivity_adjustment=0.1):
    """
    This function set off by model runs and summarise results to be used on the front end.
    It does all the calculations once an update of prices is detected.
    This is to avoid too much dynamic calculation in the frontend.
    
    TODO: Currently set as USDJPY, but for future update the ticker probably need to be dynamic.
    """
    
    print("forecast Started......")
    pred_reverse_v4, pred_continue_v4, pred_historical_v4, _ = standard_analysis(model_ticker, "v4")
    pred_reverse_v5, pred_continue_v5, pred_historical_v5, _ = standard_analysis(model_ticker,"v5")
    pred_reverse_1h_v5, pred_continue_1h_v5, pred_historical_1h_v5, _ = standard_analysis(model_ticker, "1h_v5")

    historical_headers = pred_historical_v4['pred_historical']['trade_headers']
    historical_labels = {'Periods': historical_headers}
    potential_trade_v4 = pred_historical_v4['pred_historical']['trade_breakdown']['trade_type']
    potential_trade_v5 = pred_historical_v5['pred_historical']['trade_breakdown']['trade_type']
    
    pred_collection = {
        "pred_reverse_v4": pred_reverse_v4,
        "pred_reverse_v5": pred_reverse_v5,
        "pred_reverse_1h_v5": pred_reverse_1h_v5,
        "pred_continue_v4": pred_continue_v4,
        "pred_continue_v5": pred_continue_v5,
        "pred_continue_1h_v5": pred_continue_1h_v5,
                       
                       }

    ticker_instance = get_object_or_404(Ticker, symbol=model_ticker)
    prices = Price.objects.filter(ticker=ticker_instance)
    
    
    current_time = datetime.datetime.now()
    
    rounded_time = current_time - datetime.timedelta(minutes=current_time.minute % 5,
                                                seconds=current_time.second,
                                                microseconds=current_time.microsecond)

    # Extract hour, minute, and second from the current time
  
    minute = current_time.minute
    second = current_time.second

    # Calculate the total number of seconds elapsed in the current hour
    total_seconds_in_hour = 3600  # 60 seconds * 60 minutes

    # Calculate the total number of seconds elapsed so far in the current hour
    elapsed_seconds = (minute * 60) + second

    # Calculate the percentage of the hour elapsed
    percentage_elapsed = (elapsed_seconds / total_seconds_in_hour) * 100
    
    #sort prices table in ascending so latest price on the bottom
    #note that html likes to work with array if using indexing
    prices_df = pd.DataFrame(list(prices.values()))
    sorted_prices_df = prices_df.sort_values(by='date', ascending=True)
    last_four_prices_df = sorted_prices_df.tail(4)
    open_prices = last_four_prices_df['open'].tolist()
    volume = last_four_prices_df['volume'].tolist()
    close_prices = last_four_prices_df['close'].tolist()
    open_prices_1hr = open_prices[-1]
    open_prices_4hr = open_prices[0]
    candle_size_1hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[3]['open']
    candle_size_4hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[0]['open']
    
        
    comparison_comment, send_email_enabled = compare_version_results(pred_collection, 1, 1 )
    general_ticker_info = general_ticker_results(last_four_prices_df, 1)
    
    hist_comparer = ModelComparer(pred_historical_v4, pred_historical_v5, pred_historical_1h_v5, 3, 1 )
    hist_comment = hist_comparer.comment
    hist_potential_trade = hist_comparer.trade_position
    hist_trade_target = hist_comparer.trade_target
    hist_bb_target1 = hist_comparer.bb_target1
    hist_bb_target4 = hist_comparer.bb_target4
    hist_flatness = hist_comparer.flatness
    
    cont_comparer = ModelComparer(pred_continue_v4, pred_continue_v5, pred_continue_1h_v5, 0, 1 )
    cont_comment = cont_comparer.comment
    cont_potential_trade = cont_comparer.trade_position
    cont_trade_target = cont_comparer.trade_target
    cont_bb_target1 = hist_comparer.bb_target1
    cont_bb_target4 = cont_comparer.bb_target4
    cont_flatness = cont_comparer.flatness
    
    rev_comparer = ModelComparer(pred_reverse_v4, pred_reverse_v5, pred_reverse_1h_v5, 0, 1 )
    rev_comment = rev_comparer.comment
    rev_potential_trade = rev_comparer.trade_position
    rev_trade_target = rev_comparer.trade_target
    rev_bb_target1 = hist_comparer.bb_target1
    rev_bb_target4 = rev_comparer.bb_target4
    rev_flatness = rev_comparer.flatness
    
    #calculate entry and exit point 
    trade_target = hist_trade_target
    if hist_potential_trade == 'Buy':
        entry_adjustment = -0.04
        stop_adjustment = -0.1
    else:
        entry_adjustment = 0.04
        stop_adjustment = 0.1
        trade_target = -trade_target
        
    projected_volume = volume[3] / percentage_elapsed * 100
    if projected_volume < 2000:
        exit_adjustment = 1.2
    else:
        exit_adjustment = 1
    
    entry_point = open_prices[-1] + entry_adjustment
    exit_point = open_prices[-1] + trade_target/100/exit_adjustment + entry_adjustment
    stop_loss = open_prices[-1] + stop_adjustment + entry_adjustment
    risk_reward = abs(entry_point - exit_point) / abs(entry_point - stop_loss)
    
    
    v4_pred_pl = []
    #calculate the profit or loss according to the model prediction based on open price compared to current price.
    for price, direction in zip(open_prices, potential_trade_v4):
        if direction == "Buy":
            pl = close_prices[-1] - price 
        else:
            pl = price - close_prices[-1]
    
        v4_pred_pl.append(round(pl*100))

    v5_pred_pl = []
    #calculate the profit or loss according to the predictions.
    for price, direction in zip(open_prices,potential_trade_v5):
        if direction == "Buy":
            pl = close_prices[-1] - price 
        else:
            pl = price - close_prices[-1]
    
        v5_pred_pl.append(round(pl*100))
        
        
    # sensitivity test to see how stable model is
    pred_var_pos, pred_var_neg, pos_trade, neg_trade = variability_analysis(model_ticker, 0.1)
    
    # model predicts reversal
    reverse_pred_results = standard_analysis_reverse(model_ticker, "v1_reverse")
    reverse_pred = reverse_pred_results['predictions_label']
    reverse_prob = reverse_pred_results['model_prediction_proba']*100
    reverse_prob = reverse_prob.tolist()
    
    average_open_price = sum(open_prices) / len(open_prices)
    
    
    print("Comments>>>>>>>>")
    data = {
        "comments": 
            {
                "hist": hist_comment,
                "cont": cont_comment,
                "rev": rev_comment
            },
            
        "potential_trade":
            {
                "hist": hist_potential_trade,
                "cont": cont_potential_trade,
                "rev": rev_potential_trade,
            },
        "trade_target":
            {
                "hist": hist_trade_target,
                "cont": cont_trade_target,
                "rev": rev_trade_target,
            },
        "trade_strategy":
            {
                "open_price": open_prices_1hr,
                "entry": entry_point,
                "exit": exit_point,
                "stop_loss": stop_loss,
                "risk_reward": risk_reward,
            },
        "current_market":
            {
                "hist_label": historical_labels,
                "open_prices": open_prices,
                "close_prices": close_prices,
                "volume": volume,
                "projected_volume": projected_volume,
                "average_open_price": average_open_price,
                "open_prices_1hr": open_prices_1hr,
                "open_prices_4hr": open_prices_4hr,
                "candle_size_1hr": candle_size_1hr,
                "candle_size_4hr": candle_size_4hr,
                "trade_1hr": trade_direction(candle_size_1hr),
                "trade_4hr": trade_direction(candle_size_4hr),
      
            },
        "hist_trade_outcome": #to access the model predictions
            {
                "v4_pl":  v4_pred_pl,
                "v5_pl":  v5_pred_pl,
            },
        "sensitivity_hist_model":
            {
                '10pips':{'prediction': pred_var_pos, 'trade':pos_trade},
                '-10pips':{'prediction': pred_var_neg, 'trade': neg_trade}
            },
        "reversal_model":
            {
                "reverse_pred": reverse_pred,
                "reverse_prob": reverse_prob,
                
            },
        "bb_target1":
            {
                "hist": hist_bb_target1,
                "cont": cont_bb_target1,
                "rev": rev_bb_target1,
            },
        "bb_target4":
            {
                "hist": hist_bb_target4,
                "cont": cont_bb_target4,
                "rev": rev_bb_target4,
            },
        "flatness":
            {
                "hist": hist_flatness,
                "cont": cont_flatness,
                "rev": rev_flatness,
            },
    }

    write_to_mongo(f"{model_ticker}_key_results", data)

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

        if model_version == "v4" or model_version == "v5":
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
    pos_trade = model_comparer_pos.trade_position
    neg_trade = model_comparer_neg.trade_position
    
   
    return version_comment_pos, version_comment_neg, pos_trade, neg_trade