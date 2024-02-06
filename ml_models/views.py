import pandas as pd
import os
import sys


import importlib.util
from django.shortcuts import render, get_object_or_404, HttpResponse
from .utils.trade import trade_direction
from .utils.analysis_comments import comment_model_results, compare_version_results
from .utils.predictive_analysis import standard_analysis, model_run, trade_forecast_assessment
from .utils.access_results import read_prediction_from_json
from .utils.manual_model_input import manual_price_input

from .form import ModelParameters, ModelSelection
from prices.models import Price
from tickers.models import Ticker



# Create your views here.
def ml_predictions(request):
    """
    This is a view function that pass the model predictions to the the frontend.
    Model predictions is saved as dictionary of array containing the probabilities for each profit/loss cateogires.
    """
    
    form = ModelSelection(request.POST)
    
    if request.method == 'POST' and form.is_valid():
        model_version = form.cleaned_data['model_version']
    else:
        model_version = 'v4'
        
    pred_reverse, pred_continue, pred_historical, pred_variability = standard_analysis(model_version)
    pred_reverse_v4, pred_continue_v4, pred_historical_v4, pred_variability_v4 = standard_analysis('v4')
    pred_reverse_v5, pred_continue_v5, pred_historical_v5, pred_variability_v5 = standard_analysis('v5')
    pred_reverse_1h_v5, pred_continue_1h_v5, pred_historical_1h_v5, pred_variability_1h_v5 = standard_analysis('1h_v5')
    
    

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
    version_comment, _ = compare_version_results(pred_reverse_v4, pred_reverse_v5, pred_reverse_1h_v5, last_four_prices_df, 0 )
    
    context={'form': form, 'date': date, 'pred_reverse': pred_reverse, 'pred_continue':pred_continue, 'pred_historical': pred_historical, 'pred_variability': pred_variability, 
             'open_prices': open_prices, 'close_prices': close_prices,'trade':trade, 'candle_size':candle_size, 'comment':comment, 'version_comment': version_comment}
    
    return render(request, 'ml_models/ml_predictions.html', context)

def ml_predictions2(request):
    """
    This is a view function that pass the model predictions to the the frontend.
    Model predictions is saved as dictionary of array containing the probabilities for each profit/loss cateogires.
    """
    
    form = ModelSelection(request.POST)
    
    if request.method == 'POST' and form.is_valid():
        model_version = form.cleaned_data['model_version']
    else:
        model_version = 'v4'
        
        
    pred_reverse, pred_continue, pred_historical, pred_variability = standard_analysis(model_version)
    
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
    
    context={'form': form, 'date': date, 'pred_reverse': pred_reverse, 'pred_continue':pred_continue, 'pred_historical': pred_historical, 'pred_variability': pred_variability, 
             'open_prices': open_prices, 'close_prices': close_prices,'trade':trade, 'candle_size':candle_size}
    
    return render(request, 'ml_models/ml_predictions.html', context)


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
            results, _, _, _, _ = model_run(user_input, model_version)            
     
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
    
    if request.method == 'POST':
        form = ModelSelection(request.POST)

        if form.is_valid():
            model_version = form.cleaned_data['model_version']
            model_results = trade_forecast_assessment(model_version)

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="model_results_{model_version}.csv"'
            model_results.to_csv(path_or_buf=response, index=False, encoding='utf-8')

            return response
    else:
        form = ModelSelection()

    return render(request, 'ml_models/export_model_results.html', {'form': form})
    