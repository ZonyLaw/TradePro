import pandas as pd
import os
import sys

import importlib.util
from django.contrib import messages
from django.shortcuts import render, get_object_or_404, HttpResponse
from .utils.trade import trade_direction
from .utils.analysis_comments import comment_model_results, compare_version_results
from .utils.predictive_analysis import standard_analysis, model_run, trade_forecast_assessment, variability_analysis
from .utils.access_results import read_prediction_from_json, write_to_csv
from .utils.manual_model_input import manual_price_input

from .form import ModelParameters, ModelSelection, VersionSelection
from prices.models import Price
from tickers.models import Ticker



def ml_predictions(request):
    """
    This is a view function that pass the model predictions to the the frontend.
    Model predictions is saved as dictionary of array containing the probabilities for each profit/loss cateogires.
    """
    if request.method == 'POST':
        form = VersionSelection(request.POST)
    else:
        form = VersionSelection()
    
    if form.is_valid():
        model_version = form.cleaned_data['model_version']
    else:
        model_version = 'v4'
        
    model_ticker = "USDJPY"
        
    pred_reverse, pred_continue, pred_historical, pred_variability = standard_analysis(model_ticker, model_version)
    if model_version == "v4":
        pred_reverse_v5, _, _, _  = standard_analysis(model_ticker,'v5')
        pred_reverse_1h_v5, _, _, _ = standard_analysis(model_ticker,'1h_v5')
        pred_reverse_v4 = pred_reverse
    elif model_version == "v5":
        pred_reverse_v4, _, _, _ = standard_analysis(model_ticker,'v4')
        pred_reverse_1h_v5, _, _, _  = standard_analysis(model_ticker,'1h_v5')
        pred_reverse_v5 = pred_reverse
    elif model_version == "1h_v5":
        pred_reverse_v4, _, _, _ = standard_analysis(model_ticker,'v4')
        pred_reverse_v5, _, _, _  = standard_analysis(model_ticker,'v5')
        pred_reverse_1h_v5 = pred_reverse
    
    
    print("for prediction page", pred_reverse)
    

    # pred_reverse_v4 = read_prediction_from_json(f'USDJPY_pred_reverse_v4.json')
    # pred_reverse_v5 = read_prediction_from_json(f'USDJPY_pred_reverse_v5.json')
    # pred_reverse_1h_v5 = read_prediction_from_json(f'USDJPY_pred_reverse_1h_v5.json')
    
    # pred_reverse = read_prediction_from_json(f'USDJPY_pred_reverse_{model_version}.json')
    # pred_continue = read_prediction_from_json(f'USDJPY_pred_continue_{model_version}.json')
    # pred_historical = read_prediction_from_json(f'USDJPY_pred_historical_{model_version}.json')
    # pred_variability = read_prediction_from_json(f'USDJPY_pred_variability_{model_version}.json')
    
    
    ticker_instance = get_object_or_404(Ticker, symbol="USDJPY")
    prices = Price.objects.filter(ticker=ticker_instance)
    
    #sort prices table in ascending so latest price on the bottom
    #note that html likes to work with array if using indexing
    prices_df = pd.DataFrame(list(prices.values()))
    sorted_prices_df = prices_df.sort_values(by='date', ascending=True)
    last_four_prices_df = sorted_prices_df.tail(4)
    open_prices = last_four_prices_df['open'].tolist()
    close_prices = last_four_prices_df['close'].tolist()
    
    #creating bespoke context for front-end    
    date = last_four_prices_df.iloc[3]['date']
    trade_diff_1hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[3]['open']
    trade_diff_4hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[0]['open']
    
    
    trade = {"one" :trade_direction(trade_diff_1hr),
    "four": trade_direction(trade_diff_4hr)}
    
    candle_size = {"one" :trade_diff_1hr,
    "four": trade_diff_4hr}
    
    comment = comment_model_results(pred_continue,"pred_continue")
    version_comment, _ = compare_version_results(pred_reverse_v4, pred_reverse_v5, pred_reverse_1h_v5, 0, 0 )
    
    write_to_csv(date, version_comment, "variability_results.csv")
    
    context={'form': form, 'date': date, 'pred_reverse': pred_reverse, 'pred_continue':pred_continue, 'pred_historical': pred_historical, 'pred_variability': pred_variability, 
             'open_prices': open_prices, 'close_prices': close_prices,'trade':trade, 'candle_size':candle_size, 'comment':comment, 'version_comment': version_comment}
    
    return render(request, 'ml_models/ml_predictions.html', context)


def ml_variability(request):
    """
    return:
    This returns the variability_results for all 3 models.
    """
    
    ticker_instance = get_object_or_404(Ticker, symbol="USDJPY")
    prices = Price.objects.filter(ticker=ticker_instance)
    
    #sort prices table in ascending so latest price on the bottom
    #note that html likes to work with array if using indexing
    prices_df = pd.DataFrame(list(prices.values()))
    sorted_prices_df = prices_df.sort_values(by='date', ascending=True)
    last_four_prices_df = sorted_prices_df.tail(4)
    
    version_comment_pos, version_comment_neg = variability_analysis(ticker_instance.symbol)
    
    context = {'version_comment_pos': version_comment_pos, 'version_comment_neg': version_comment_neg }
    
    return render(request, 'ml_models/ml_variability.html', context)


def ml_report(request):
    """
    This is a view function that pass the model predictions to the the frontend.
    Model predictions is saved as dictionary of array containing the probabilities for each profit/loss cateogires.
    """
    if request.method == 'POST':
        form = ModelSelection(request.POST)
    else:
        form = ModelSelection()
        
    if form.is_valid():
        model_ticker = form.cleaned_data['ticker']
    else:
        model_ticker = 'USDJPY'
    
    ticker_instance = get_object_or_404(Ticker, symbol=model_ticker)
    prices = Price.objects.filter(ticker=ticker_instance)
    
    #sort prices table in ascending so latest price on the bottom
    #note that html likes to work with array if using indexing
    prices_df = pd.DataFrame(list(prices.values()))
    sorted_prices_df = prices_df.sort_values(by='date', ascending=True)
    last_four_prices_df = sorted_prices_df.tail(4)
    date = last_four_prices_df.iloc[3]['date']
    open_prices = last_four_prices_df['open'].tolist()
    close_prices = last_four_prices_df['close'].tolist()
    volume = last_four_prices_df['volume'].tolist()
    
    trade_diff_1hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[3]['open']
    trade_diff_4hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[0]['open']
    candle_size = {"one" :trade_diff_1hr,
        "four": trade_diff_4hr}
    
    trade = {"one" :trade_direction(trade_diff_1hr),
        "four": trade_direction(trade_diff_4hr)}
    
    #retrieve saved results from last calculation performed by updater.py
    pred_historical_v4 = read_prediction_from_json(model_ticker, f'USDJPY_pred_historical_v4.json')
    pred_historical_v5 = read_prediction_from_json(model_ticker, f'USDJPY_pred_historical_v5.json')
    pred_historical_1h_v5 = read_prediction_from_json(model_ticker, f'USDJPY_pred_historical_1h_v5.json')
    
    #extracting final results
    historical_headers = pred_historical_v4['pred_historical'][1]['heading']
    potential_trade_results_v4 = pred_historical_v4['pred_historical'][2]['item']['Potential Trade']
    potential_trade_results_v5 = pred_historical_v5['pred_historical'][2]['item']['Potential Trade']
    potential_trade_results_1h_v5 = pred_historical_1h_v5['pred_historical'][2]['item']['Potential Trade']
    split_potential_trade_result = [trade.split(':') for trade in potential_trade_results_v5]
    trade_target = float(split_potential_trade_result[-1][-1].strip())
    
    #save historical array as a dictionary for frontend access
    historical_labels = {'Periods': historical_headers}
    historical_trade_results = {
    'v4': potential_trade_results_v4,
    'v5': potential_trade_results_v5,
    '1h_v5': potential_trade_results_1h_v5
    }
        
    potential_trade = pred_historical_v5['pred_historical'][2]['item']['Potential Trade'][3]
    
    #calculate entry and exit point
    if trade_target > 0:
        entry_adjustment = -0.04
        stop_adjustment = -0.1
    else:
        entry_adjustment = 0.04
        stop_adjustment = 0.1
        
    entry_point = open_prices[-1] + entry_adjustment
    exit_point = open_prices[-1] + trade_target/100 + entry_adjustment
    stop_loss = open_prices[-1] + stop_adjustment + entry_adjustment

    #sensitivity test save as dictionary for front-end access
    pred_var_pos, pred_var_neg = variability_analysis(model_ticker)
    pred_var_list = {
        '10 pips':pred_var_pos,
        '-10 pips':pred_var_neg,
    }
    
    
    #retrieve saved results from last calculation performed by updater.py
    pred_reverse_v4 = read_prediction_from_json(model_ticker, f'USDJPY_pred_reverse_v4.json')
    pred_reverse_v5 = read_prediction_from_json(model_ticker, f'USDJPY_pred_reverse_v5.json')
    pred_reverse_1h_v5 = read_prediction_from_json(model_ticker, f'USDJPY_pred_reverse_1h_v5.json')
    
    #extracting final results
    reverse_headers = pred_reverse_v4['pred_reverse'][1]['heading']
    reverse_trade_results_v4 = pred_reverse_v4['pred_reverse'][2]['item']['Potential Trade']
    reverse_trade_results_v5 = pred_reverse_v5['pred_reverse'][2]['item']['Potential Trade']
    reverse_trade_results_1h_v5 = pred_reverse_1h_v5['pred_reverse'][2]['item']['Potential Trade']
    
    #save array of reversed results as a dictionary for frontend access
    reverse_labels = {'Periods': reverse_headers}
    reverse_trade_lists = {
    'v4': reverse_trade_results_v4,
    'v5': reverse_trade_results_v5,
    '1h_v5': reverse_trade_results_1h_v5
    }

    version_comment, _ = compare_version_results(pred_historical_v4, pred_historical_v5, pred_historical_1h_v5, 3, 1 )
    
    # for model_version in model_versions:
    #     globals()[f'pred_historical_{model_version}'] \
    #         = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_reverse_{model_version}.json')


    context={'form': form,  'date': date, 'candle_size':candle_size, 'trade': trade, 'version_comment':version_comment,
             'open_prices': open_prices, 'close_prices': close_prices, 'volume': volume,
             'entry_point': entry_point, 'exit_point': exit_point, 'stop_loss': stop_loss, 'potential_trade': potential_trade, 
             'historical_labels': historical_labels, 'historical_trade_results': historical_trade_results,
             'pred_var_list': pred_var_list,
             'reverse_labels': reverse_labels, 'reverse_trade_results': reverse_trade_lists,}
    
    return render(request, 'ml_models/ml_report.html', context)


def ml_manual(request):
    
    """
    This is a function to get user input for running the model manually.
    The results are displayed in the same webpage as the input form.
    User can update input to get new results.
    NOTE: the results are slightly different becuase the 4hr close is assumed to be same as 1hr close.
    
    """
    
    # Dynamically get the module path involves defining the parent directory
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Add the parent directory to the Python path
    sys.path.append(parent_directory)
    module_name = f'ml_models.trained_models.USDJPY.pl_predictions.v4.data_processing'
    print("Module Path:", os.path.join(parent_directory, module_name.replace('.', os.sep) + ".py"))
    try:
        dp = importlib.import_module(module_name)
    except ImportError:
        print(f"Error importing data_processing module for model_version: v4")
        dp = None
    
    
    form = ModelParameters(request.POST)

    
    ticker_instance = get_object_or_404(Ticker, symbol="USDJPY")
    prices = Price.objects.filter(ticker=ticker_instance)
    
    prices_df = pd.DataFrame(list(prices.values()))
    prices_df = prices_df.sort_values(by='date', ascending=True)
    # last_prices_df = prices_df.tail(2)
    last_price_stats = dp.stats_df_gen(prices_df,2)
    
    results = []
    if request.method == 'POST':
        form = ModelParameters(request.POST)
        if form.is_valid():
            model_version = form.cleaned_data['model_version']
            user_input = manual_price_input(form)
            results, _, _, _, _ = model_run("USDJPY", user_input, model_version)            
     
    else:
           
        # Initialize the form with default values
        form = ModelParameters(initial={           
            'model_version': "v4",
            #note current prices are used 
            'open': last_price_stats['open'].values[1],
            'close': last_price_stats['close'].values[1],
            'open_lag1': last_price_stats['open'].values[0],
            'close_lag1': last_price_stats['close'].values[0],
            'ma50': last_price_stats['ma50_1'].values[1],
            
            # note we just take the current close as four close but is different in actual fact.
            'close_4': last_price_stats['close'].values[1],
            
            #the rest of the indicator is lagged
            'ma20_4': last_price_stats['ma20_4'].values[1],
            'ma50_4': last_price_stats['ma50_4'].values[1],
            'ma100_4': last_price_stats['ma100_4'].values[1],
            'bb20_high': last_price_stats['upper_bb20_1'].values[1],
            'bb20_low': last_price_stats['lower_bb20_1'].values[1],
            'bb20_high_4': last_price_stats['upper_bb20_4'].values[1],
            'bb20_low_4': last_price_stats['lower_bb20_4'].values[1],
            'hour': last_price_stats['hr'].values[1],
            'trend_strength_1': last_price_stats['trend_strength_1'].values[1],
            'bb_status_1': "inside_bb",

        })

    context = {'form':form, 'results':results}
    
    return render(request, 'ml_models/ml_manual_analysis.html', context)


def export_model_results(request):
    
    messages.clear(request)
    
    if request.method == 'POST':
        form = VersionSelection(request.POST)

        if form.is_valid():
            model_version = form.cleaned_data['model_version']
            model_results = trade_forecast_assessment(model_version)

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="model_results_{model_version}.csv"'
            model_results.to_csv(path_or_buf=response, index=False, encoding='utf-8')

            return response
    else:
        form = VersionSelection()

    return render(request, 'ml_models/export_model_results.html', {'form': form})
    
    
def delete_file(request, filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Assuming this view is in a module
    base_dir_up_one_levels = os.path.abspath(os.path.join(base_dir, os.pardir))
    file_path = os.path.join(base_dir_up_one_levels, 'media', 'model_results', filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return HttpResponse("File deleted successfully.")
    else:
        return HttpResponse("File does not exist.")
    

def export_file(request, filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Assuming this view is in a module
    base_dir_up_one_levels = os.path.abspath(os.path.join(base_dir, os.pardir))
    file_path = os.path.join(base_dir_up_one_levels, 'media', 'model_results', filename)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = 'attachment; filename="{}"'.format(filename)
        return response
    else:
        return HttpResponse("File does not exist.")