import os
import joblib
import pandas as pd
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
    print(X_live_subset)
    # predict the probability
    model_prediction_proba = model_pipeline.predict_proba(X_live_subset)
    
    
    # Transform predicted probabilities to labels using label mapping
    # model_prediction_labels = model_label_map.inverse_transform(model_prediction_proba.argmax(axis=1))
    
    model_prediction_labels = model_prediction_proba.argmax(axis=1)

    # Compare predictions with actual outcomes
    comparison_results = {
        'actual_labels': y_actual,
        'predicted_labels': model_prediction_labels,
    }
    
    
    
    # Compare predictions with actual outcomes
    accuracy = calculate_accuracy(model_prediction_labels, y_actual)
    print("comparisoin>>>>>>>>>>>>>>>>>>>>>>>", comparison_results)
    print("Accuracy", accuracy)
    
    
    result_dict = {}
    #First loop goes through the probability profit/loss categories
    #Second loop goes through the array size of the probability dictionary
    for i in range(6):
            result_dict.update({
                f"{model_label_map[i]}": [
                    round(model_prediction_proba[j, i] * 100, 2) 
                    for j in range(model_prediction_proba.shape[0])
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
    
    print("predict profit model>>>>", model_prediction_proba[0,1])
    print("result_dict.....", result_dict)
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