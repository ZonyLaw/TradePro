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


class ModelResultsFormatter:
    def __init__(self, pred_name, heading_labels):
        self.pred_name = pred_name
        self.heading_labels = heading_labels

    def transform_format(self, main_results, extra_results):
        current_timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        transformed_data = {
            self.pred_name: [
                {"date": current_timestamp},
                {"heading": self.heading_labels},
                {"item": main_results},
                {"extra": extra_results}
            ]
        }
        return transformed_data

    def format_model_results(self, model_prediction_proba, model_prediction, model_label_map):
        result_dict = {}
        for i, label in enumerate(model_label_map):
            result_dict[label] = [round(proba * 100, 2) for proba in model_prediction_proba[:, i]]

        potential_trade = []
        trade_type = []
        trade_target = []

        for prediction in model_prediction:
            direction = "Sell" if prediction < 3 else "Buy"
            profit_label = model_label_map[prediction]
            potential_trade.append(f"{direction} target: {profit_label}")
            trade_type.append(direction)
            trade_target.append(profit_label)

        result_dict["Potential Trade"] = potential_trade
        extra_results = {
            "Trade Type": trade_type,
            "Trade Target": trade_target
        }

        return result_dict, extra_results



def load_file(file_path):
    #data management-loads the filepath to get the model
    return joblib.load(filename=file_path)


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
    
    # extract the relevant subset features related to this pipeline
    X_live_subset = X_live_discretized.filter(model_features)
    
    # predict the probability for each of the cateogries
    model_prediction_proba = model_pipeline.predict_proba(X_live_subset)
    
    #prediction of model by getting the category with the maximum probability
    model_prediction = model_prediction_proba.argmax(axis=1)

    return {
        "model_prediction_proba": model_prediction_proba,
        "model_prediction": model_prediction,
        "model_labels_map": model_labels_map,
        "X_live_discretized": X_live_discretized
    }


# Create an instance of ModelResultsFormatter
formatter = ModelResultsFormatter(pred_name="Prediction", heading_labels=["Label1", "Label2"])

# Run model
model_results = model_run(ticker, X_live, model_version)

# Format results using the formatter instance
formatted_results = formatter.format_model_results(**model_results)