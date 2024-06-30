import os
import sys
import joblib
import importlib.util
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
import numpy as np


def load_file(file_path):
    #data management-loads the filepath to get the model
    return joblib.load(filename=file_path)


def news_model_run(ticker, X_live, model_version):
    
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
        f"trained_models/{ticker}/pl_predictions/{version}/news_pipeline_model.pkl")
    reverse_pipeline_data_cleaning = load_file(
        f"trained_models/{ticker}/pl_predictions/{version}/news_pipeline_data_cleaning.pkl")
    model_features = (pd.read_csv(f"trained_models/{ticker}/pl_predictions/{version}/X_test.csv")
                       .columns
                       .to_list()
                       )
    label_map = {"0":"Sell", "1":"Buy"}
    
    # Discretize the target variable (ie. y dependent) 
    
    y_test_headers = (pd.read_csv(f"trained_models/{ticker}/pl_predictions/{version}/y_test.csv")
                       .columns
                       .to_list()
                       )
    
    # extract the relevant subset features related to this pipeline
    X_live_subset = X_live.filter(model_features)
    X_live_subset= reverse_pipeline_data_cleaning.transform(X_live_subset)
    
    # predict the probability for each of the cateogries
    model_prediction = model_pipeline.predict(X_live_subset)
    model_prediction_proba = np.amax(model_pipeline.predict_proba(X_live_subset), axis=1)
    predictions_label = [label_map[str(int(prediction))] for prediction in model_prediction]
    
    #the dictionary is kept the same as trade model with additional prediction label. This is to make sure it works consistently with the code.
    return {
            'model_prediction_proba': model_prediction_proba,
            'model_prediction': model_prediction,
            'model_labels_map': ['Sell','Buy'],
            'X_live_discretized': X_live,
            'predictions_label': predictions_label
        }


def standard_analysis_news(ticker, model_version, sensitivity_adjustment=0.1):
    
    """
    This is a function to generate some standard analysis to show on the webpage.
    Pre-defined scenarios are inputted into the model.
    This calls on the model_run function which pulls all relevant inputs to generate results.
    Json file will be produced saving the results.
    Returns:
        dictionary: returns results from model for the different scenarios
    """
       
    if model_version == "v1_reverse1":
        dp = v4Processing(ticker=ticker)
    else:
        dp = StandardPriceProcessing(ticker=ticker)
    
    X_live_reverse = dp.historical_record(4)
    
    
    pred_reverse_results = reverse_model_run(ticker, X_live_reverse, model_version)

     
    return pred_reverse_results

