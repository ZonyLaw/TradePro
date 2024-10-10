import pandas as pd
import os
import sys
import importlib.util

import datetime
import pytz
import zoneinfo

from django.contrib import messages
from django.shortcuts import render, get_object_or_404, HttpResponse
from .utils.trade import trade_direction
from .utils.analysis_comments import comment_model_results, compare_version_results, ModelComparer
from .utils.trade_model import standard_analysis, model_run, trade_forecast_assessment, variability_analysis
from .utils.access_results import read_prediction_from_json, write_to_csv, read_prediction_from_Mongo
from .utils.manual_model_input import manual_price_input, news_param_input

from .form import ModelParameters, NewsParameters, ModelSelection, VersionSelection, VariabilitySize
from prices.models import Price
from tickers.models import Ticker
from users.models import Profile
from ml_models.utils.bespoke_model import v4Processing
from ml_models.utils.price_processing import StandardPriceProcessing

from ml_models.utils.reverse_model import standard_analysis_reverse
from ml_models.utils.news_model import news_model_run


def about(request):
      return render(request, 'ml_models/about.html')


def ml_variability(request):
    ticker_instance = get_object_or_404(Ticker, symbol="USDJPY")
    prices = Price.objects.filter(ticker=ticker_instance)
    
    # Sort prices table in ascending order so the latest price is at the bottom
    prices_df = pd.DataFrame(list(prices.values()))
    sorted_prices_df = prices_df.sort_values(by='date', ascending=True)
    last_four_prices_df = sorted_prices_df.tail(4)
    
    if request.method == 'POST':
        
        form = VariabilitySize(request.POST)
        if form.is_valid():
            sensitivity_adjustment = form.cleaned_data['sensitivity_adjustment']
            version_comment_pos, version_comment_neg = variability_analysis(ticker_instance.symbol, sensitivity_adjustment/100)
            context = {'adjustment': sensitivity_adjustment, 'version_comment_pos': version_comment_pos, 'version_comment_neg': version_comment_neg, 'form': form}
            return render(request, 'ml_models/ml_variability.html', context)
    else:
        form = VariabilitySize()
    
    context = {'form': form}
    return render(request, 'ml_models/ml_variability.html', context)


def ml_report_backup(request):
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
    
    # model_version = "v4"
    # if model_version == "v4":
    #     dp = v4Processing(ticker=model_ticker)
    # else:
    #     dp = StandardPriceProcessing(ticker=model_ticker)
    
    # X_live = dp.historical_record(4)
    
    # bb_status = X_live['bb_status_1'].tolist()

    
    #sort prices table in ascending so latest price on the bottom
    #note that html likes to work with array if using indexing
    prices_df = pd.DataFrame(list(prices.values()))
    sorted_prices_df = prices_df.sort_values(by='date', ascending=True)
    last_four_prices_df = sorted_prices_df.tail(4)
    date = last_four_prices_df.iloc[3]['date']
    open_prices = last_four_prices_df['open'].tolist()
    close_prices = last_four_prices_df['close'].tolist()

    
    #looking at the 1hr and 4hr candle sticks direction
    trade_diff_1hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[3]['open']
    trade_diff_4hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[0]['open']
    candle_size = {"one" :trade_diff_1hr,
        "four": trade_diff_4hr}
    
    trade = {"one": trade_direction(trade_diff_1hr),
        "four": trade_direction(trade_diff_4hr)}
    
    # Iterate through the last 4 rows of the DataFrame
    trade_dict = []
    for i in range(0, len(last_four_prices_df), 1):
        trade_diff = last_four_prices_df.iloc[i]['close'] - last_four_prices_df.iloc[i]['open']
        if trade_diff > 0:
            trade_dict.append('Buy')
        elif trade_diff < 0:
            trade_dict.append('Sell')
    
    #retrieve saved results from last calculation performed by updater.py
    pred_historical_v4 = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_historical_v4.json')
    pred_historical_v5 = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_historical_v5.json')
    pred_historical_1h_v5 = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_historical_1h_v5.json')
    
    #extracting final results
    historical_headers = pred_historical_v4['pred_historical'][1]['trade_headers']
    potential_trade_results_v4 = pred_historical_v4['pred_historical'][2]['trade_results']['Potential Trade']
    potential_trade_results_v5 = pred_historical_v5['pred_historical'][2]['trade_results']['Potential Trade']
    potential_trade_results_1h_v5 = pred_historical_1h_v5['pred_historical'][2]['trade_results']['Potential Trade']
 
    potential_trade_label_v4 = pred_historical_v4['pred_historical'][3]['trade_breakdown']['trade_type']
    potential_trade_label_v5 = pred_historical_v5['pred_historical'][3]['trade_breakdown']['trade_type']
    
    #save historical array as a dictionary for frontend access
    historical_labels = {'Periods': historical_headers}
    historical_trade_results = {
    'v4': potential_trade_results_v4,
    'v5': potential_trade_results_v5,
    '1h_v5': potential_trade_results_1h_v5
    }
        

    #retrieve saved results from last calculation performed by updater.py
    pred_reverse_v4 = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_reverse_v4.json')
    pred_reverse_v5 = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_reverse_v5.json')
    pred_reverse_1h_v5 = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_reverse_1h_v5.json')
    
    #extracting final results
    reverse_headers = pred_reverse_v4['pred_reverse'][1]['trade_headers']
    reverse_trade_results_v4 = pred_reverse_v4['pred_reverse'][2]['trade_results']['Potential Trade']
    reverse_trade_results_v5 = pred_reverse_v5['pred_reverse'][2]['trade_results']['Potential Trade']
    reverse_trade_results_1h_v5 = pred_reverse_1h_v5['pred_reverse'][2]['trade_results']['Potential Trade']
    
    #save array of reversed results as a dictionary for frontend access
    reverse_labels = {'Periods': reverse_headers}
    reverse_trade_lists = {
    'v4': reverse_trade_results_v4,
    'v5': reverse_trade_results_v5,
    '1h_v5': reverse_trade_results_1h_v5
    }
    
    
    #retrieve saved results from last calculation performed by updater.py
    pred_continue_v4 = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_continue_v4.json')
    pred_continue_v5 = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_continue_v5.json')
    pred_continue_1h_v5 = read_prediction_from_json(model_ticker, f'{model_ticker}_pred_continue_1h_v5.json')
    
    #extracting final results
    continue_headers = pred_continue_v4['pred_continue'][1]['trade_headers']
    continue_trade_results_v4 = pred_continue_v4['pred_continue'][2]['trade_results']['Potential Trade']
    continue_trade_results_v5 = pred_continue_v5['pred_continue'][2]['trade_results']['Potential Trade']
    continue_trade_results_1h_v5 = pred_continue_1h_v5['pred_continue'][2]['trade_results']['Potential Trade']
    
    #save array of continued results as a dictionary for frontend access
    continue_labels = {'Periods': continue_headers}
    continue_trade_lists = {
    'v4': continue_trade_results_v4,
    'v5': continue_trade_results_v5,
    '1h_v5': continue_trade_results_1h_v5
    }
    
      

    current_time = datetime.datetime.now()
    
    rounded_time = current_time - datetime.timedelta(minutes=current_time.minute % 5,
                                                seconds=current_time.second,
                                                microseconds=current_time.microsecond)

    
    v4_pred_pl = []
    #calculate the profit or loss according to the predictions.
    for price, direction in zip(open_prices,potential_trade_label_v4):
        if direction == "Buy":
            pl = close_prices[-1] - price 
        else:
            pl = price - close_prices[-1]
    
        v4_pred_pl.append(round(pl*100))
        
    v5_pred_pl = []
    #calculate the profit or loss according to the predictions.
    for price, direction in zip(open_prices,potential_trade_label_v5):
        if direction == "Buy":
            pl = close_prices[-1] - price 
        else:
            pl = price - close_prices[-1]
    
        v5_pred_pl.append(round(pl*100))



    #sensitivity test save as dictionary for front-end access
    pred_var_pos, pred_var_neg = variability_analysis(model_ticker, 0.1)
    pred_var_list = {
        '10 pips':pred_var_pos,
        '-10 pips':pred_var_neg,
    }
    


    reverse_pred_results = standard_analysis_reverse("USDJPY", "v1_reverse")
    reverse_pred = reverse_pred_results['predictions_label']
    reverse_prob = reverse_pred_results['model_prediction_proba']*100
    
    context={'form': form,  'date': date, 'rounded_time': rounded_time, 'candle_size':candle_size, 
             'trade': trade, 'trade_dict':trade_dict,
      
             'historical_labels': historical_labels, 'historical_trade_results': historical_trade_results,
             'v4_pred_pl': v4_pred_pl, 'v5_pred_pl': v5_pred_pl,'pred_var_list': pred_var_list, 
             'reverse_labels': reverse_labels, 'reverse_trade_results': reverse_trade_lists,
             'continue_labels': continue_labels, 'continue_trade_results': continue_trade_lists,
             "reverse_pred": reverse_pred, "reverse_prob": reverse_prob}
    
    return render(request, 'ml_models/ml_report.html', context)


def ml_report2(request):
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
    
    # model_version = "v4"
    # if model_version == "v4":
    #     dp = v4Processing(ticker=model_ticker)
    # else:
    #     dp = StandardPriceProcessing(ticker=model_ticker)
    
    # X_live = dp.historical_record(4)
    
    # bb_status = X_live['bb_status_1'].tolist()

    
    #sort prices table in ascending so latest price on the bottom
    #note that html likes to work with array if using indexing
    prices_df = pd.DataFrame(list(prices.values()))
    sorted_prices_df = prices_df.sort_values(by='date', ascending=True)
    last_four_prices_df = sorted_prices_df.tail(4)
    # date = last_four_prices_df.iloc[3]['date']
    open_prices = last_four_prices_df['open'].tolist()
    close_prices = last_four_prices_df['close'].tolist()
    volume = last_four_prices_df['volume'].tolist()
    

  
    
    # Iterate through the last 4 rows of the DataFrame
    trade_dict = []
    for i in range(0, len(last_four_prices_df), 1):
        trade_diff = last_four_prices_df.iloc[i]['close'] - last_four_prices_df.iloc[i]['open']
        if trade_diff > 0:
            trade_dict.append('Buy')
        elif trade_diff < 0:
            trade_dict.append('Sell')
    
    key_results = read_prediction_from_Mongo(f'{model_ticker}_key_results')
    
    
    #retrieve saved results from last calculation performed by updater.py
    pred_historical_v4 = read_prediction_from_Mongo(f'{model_ticker}_pred_historical_v4')
    pred_historical_v5 = read_prediction_from_Mongo(f'{model_ticker}_pred_historical_v5')
    pred_historical_1h_v5 = read_prediction_from_Mongo(f'{model_ticker}_pred_historical_1h_v5')
    
    
    # user = request.user

    # if user.is_authenticated:  # Check if the user is authenticated
    #     try:
    #         # Try to get the user's profile
    #         profile = Profile.objects.get(user=user)
    #         # Use the user's timezone if available
    #         user_timezone = profile.timezone
    #     except Profile.DoesNotExist:
    #         # If no profile exists, fallback to UTC
    #         user_timezone = zoneinfo.ZoneInfo("UTC")
    # else:
    #     # Handle anonymous user by using a default timezone
    #     user_timezone = zoneinfo.ZoneInfo("UTC")

    # # Extract the timezone string
    # user_timezone = str(user_timezone)  
    
    user_timezone = "GB"
    
    # Parse the original date string
    original_date_str = pred_historical_v4['pred_historical']['date']
    original_date = datetime.datetime.strptime(original_date_str, "%d-%m-%Y %H:%M:%S")
    # Get the user's timezone, default to UTC if not set
    original_date = original_date.replace(tzinfo=pytz.utc)
    # user_timezone = "UTC"
    user_tz = pytz.timezone(user_timezone)
    # Convert original date to the user's timezone
    date = original_date.astimezone(user_tz)
    
    # formatted_date = date.strftime("%d-%m-%Y %H:%M:%S")
    
    #extracting final results
    historical_headers = pred_historical_v4['pred_historical']['trade_headers']
    potential_trade_results_v4 = pred_historical_v4['pred_historical']['trade_results']['Potential Trade']
    potential_trade_results_v5 = pred_historical_v5['pred_historical']['trade_results']['Potential Trade']
    potential_trade_results_1h_v5 = pred_historical_1h_v5['pred_historical']['trade_results']['Potential Trade']
 
    potential_trade_label_v4 = pred_historical_v4['pred_historical']['trade_breakdown']['trade_type']
    potential_trade_label_v5 = pred_historical_v5['pred_historical']['trade_breakdown']['trade_type']
    
    #save historical array as a dictionary for frontend access
    historical_labels = {'Periods': historical_headers}
    historical_trade_results = {
    '4hr Proj (v4)': potential_trade_results_v4,
    '4hr Proj (v5)': potential_trade_results_v5,
    '1hr Proj (1h_v5)': potential_trade_results_1h_v5
    }
        

    #retrieve saved results from last calculation performed by updater.py
    pred_reverse_v4 = read_prediction_from_Mongo(f'{model_ticker}_pred_reverse_v4')
    pred_reverse_v5 = read_prediction_from_Mongo(f'{model_ticker}_pred_reverse_v5')
    pred_reverse_1h_v5 = read_prediction_from_Mongo(f'{model_ticker}_pred_reverse_1h_v5')
    
    #extracting final results
    reverse_headers = pred_reverse_v4['pred_reverse']['trade_headers']
    reverse_trade_results_v4 = pred_reverse_v4['pred_reverse']['trade_results']['Potential Trade']
    reverse_trade_results_v5 = pred_reverse_v5['pred_reverse']['trade_results']['Potential Trade']
    reverse_trade_results_1h_v5 = pred_reverse_1h_v5['pred_reverse']['trade_results']['Potential Trade']
    
    #save array of reversed results as a dictionary for frontend access
    reverse_labels = {'Periods': reverse_headers}
    reverse_trade_lists = {
    'v4': reverse_trade_results_v4,
    'v5': reverse_trade_results_v5,
    '1h_v5': reverse_trade_results_1h_v5
    }
    
    #retrieve saved results from last calculation performed by updater.py
    pred_continue_v4 = read_prediction_from_Mongo(f'{model_ticker}_pred_continue_v4')
    pred_continue_v5 = read_prediction_from_Mongo(f'{model_ticker}_pred_continue_v5')
    pred_continue_1h_v5 = read_prediction_from_Mongo(f'{model_ticker}_pred_continue_1h_v5')
    
    #extracting final results
    continue_headers = pred_continue_v4['pred_continue']['trade_headers']
    continue_trade_results_v4 = pred_continue_v4['pred_continue']['trade_results']['Potential Trade']
    continue_trade_results_v5 = pred_continue_v5['pred_continue']['trade_results']['Potential Trade']
    continue_trade_results_1h_v5 = pred_continue_1h_v5['pred_continue']['trade_results']['Potential Trade']
    
    #save array of continued results as a dictionary for frontend access
    continue_labels = {'Periods': continue_headers}
    continue_trade_lists = {
    'v4': continue_trade_results_v4,
    'v5': continue_trade_results_v5,
    '1h_v5': continue_trade_results_1h_v5
    }
       
    
    
    current_time = datetime.datetime.now()
    
    rounded_time = current_time - datetime.timedelta(minutes=current_time.minute % 5,
                                                seconds=current_time.second,
                                                microseconds=current_time.microsecond)


    average_open_price = sum(open_prices) / len(open_prices)
    if key_results['potential_trade']['hist'] == "Buy":
        final_exit_price = min(open_prices)  - 0.2
    else:
        final_exit_price = max(open_prices) + 0.2
        
    reverse_pred_results = standard_analysis_reverse("USDJPY", "v1_reverse")
    reverse_pred = reverse_pred_results['predictions_label']
    reverse_prob = reverse_pred_results['model_prediction_proba']*100
    
    context={'form': form,  'date': date, 'rounded_time': rounded_time,  'trade_dict':trade_dict,
             'historical_labels': historical_labels, 'historical_trade_results': historical_trade_results,
             'average_open_price': average_open_price, 'final_exit_price': final_exit_price,
             'reverse_labels': reverse_labels, 'reverse_trade_results': reverse_trade_lists,
             'continue_labels': continue_labels, 'continue_trade_results': continue_trade_lists,
             "reverse_pred": reverse_pred, "reverse_prob": reverse_prob, "key_results": key_results}
    
    return render(request, 'ml_models/ml_report.html', context)


def ml_manual(request):
    
    """
    This is a function to get user input for running the model manually.
    The results are displayed in the same webpage as the input form.
    User can update input to get new results.
    NOTE: the results are slightly different becuase the 4hr close is assumed to be same as 1hr close.
    
    """
    
    model_version = "v4"
    if model_version == "v4":
        dp = v4Processing("USDJPY")
    else:
        dp = StandardPriceProcessing("USDJPY")
    
    
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


def ml_news_model_backup(request):
    
    """
    This is a function to run the news model
    
    """
    model_version = "v1_news"
    ticker_input= "USDJPY"
    form = NewsParameters(request.POST)
    
    dp = v4Processing("USDJPY")
    ticker_instance = get_object_or_404(Ticker, symbol="USDJPY")
    prices = Price.objects.filter(ticker=ticker_instance)
    
    prices_df = pd.DataFrame(list(prices.values()))
    prices_df = prices_df.sort_values(by='date', ascending=True)
    # last_prices_df = prices_df.tail(2)
    last_price_stats = dp.stats_df_gen(prices_df,2)
    
    print(last_price_stats['hr'])
    print(last_price_stats['weekday'])
    print(last_price_stats['bb_status_1'])
    
    news_pred_str = ""
    news_pred = -99
    news_prob = 0
    if request.method == 'POST':
        form = NewsParameters(request.POST)
        if form.is_valid():

            user_input = news_param_input(form)
            print(user_input)
            news_pred_results = news_model_run(ticker_input, user_input, model_version)   
            news_pred = news_pred_results['model_prediction']
            news_prob = news_pred_results['model_prediction_proba']*100 
            
            if news_pred == 0:
                news_pred_str = "Sell"
            else:
                news_pred_str = "Buy"
     
    else:
    
        form = NewsParameters(initial={     
            'weekday': last_price_stats['weekday'].values[1],
            'hour': last_price_stats['hr'].values[1], 
            'bb_status_1': last_price_stats['bb_status_1'].values[1], 
            'event': "CPI",
            'output': "better",

        })
        
        

    context = {'form':form, 'news_pred_str':news_pred_str, 'news_prob': news_prob}
         
    return render(request, 'ml_models/ml_news_model.html', context)


def ml_news_model(request):
    """
    This is a function to run the news model.
    """
    model_version = "v1_news"
    ticker_input = "USDJPY"
    form = NewsParameters(request.POST)
    
    dp = v4Processing(ticker_input)
    ticker_instance = get_object_or_404(Ticker, symbol=ticker_input)
    prices = Price.objects.filter(ticker=ticker_instance)
    
    prices_df = pd.DataFrame(list(prices.values()))
    prices_df = prices_df.sort_values(by='date', ascending=True)
    last_price_stats = dp.stats_df_gen(prices_df, 2)
    
      
    news_pred_str = {}
    news_prob_list = {}
    news_pred = -99
    news_prob = 0
    
    bb_status_list = [
        'inside_bb',
        'upper_near',
        'upper_crossover', 
        'upper_hit',
        'lower_hit', 
        'lower_near', 
        'lower_crossover',
        ]
    
    if request.method == 'POST':
        form = NewsParameters(request.POST)
        if form.is_valid():
            user_input = news_param_input(form)
            print(user_input)
            
            for bb_status in bb_status_list:
                user_input['bb_status_1'] = bb_status
                news_pred_results = news_model_run(ticker_input, user_input, model_version)
                news_pred = news_pred_results['model_prediction']
                news_prob = news_pred_results['model_prediction_proba'] * 100
                news_prob_list[bb_status] = news_prob
                
                if news_pred == 0:
                    news_pred_str[bb_status] = "Sell"
                else:
                    news_pred_str[bb_status] = "Buy"
                
    else:
        form = NewsParameters(initial={
            'weekday': last_price_stats['weekday'].values[1],
            'hour': last_price_stats['hr'].values[1],
          #  'bb_status_1': last_price_stats['bb_status_1'].values[1],
            'event': "CPI",
            'output': "better",
        })
        
    print(news_pred_str)
    context = {'form': form, 'news_pred_str': news_pred_str, 'news_prob': news_prob}
    return render(request, 'ml_models/ml_news_model.html', context)


def export_model_results(request):
    
    # messages.clear(request)
    
    if request.method == 'POST':
        form = VersionSelection(request.POST)

        if form.is_valid():
            model_version = form.cleaned_data['model_version']
            ticker = form.cleaned_data['ticker']
            model_results, _, _, _ = trade_forecast_assessment(ticker, model_version)

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="model_results_{model_version}.csv"'
            model_results.to_csv(path_or_buf=response, index=False, encoding='utf-8')

            return response
    else:
        form = VersionSelection()

    return render(request, 'ml_models/export_model_results.html', {'form': form})
    
def display_model_accuracy(request):
    
    profit_accuracy = None
    trade_accuracy = None
    buy_sell_split = {}
    
    if request.method == 'POST':
        form = VersionSelection(request.POST)

        if form.is_valid():
            model_version = form.cleaned_data['model_version']
            ticker = form.cleaned_data['ticker']
            model_results, profit_accuracy, trade_accuracy, buy_sell_split = trade_forecast_assessment(ticker, model_version)
            profit_accuracy = profit_accuracy * 100
            trade_accuracy = trade_accuracy * 100

    else:
        form = VersionSelection()
        
    context = {'form': form, "profit_accuracy": profit_accuracy, "trade_accuracy": trade_accuracy, "buy_sell_split": buy_sell_split}

    return render(request, 'ml_models/model_accuracy.html', context )

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
    
    
def notes(request):
    return render(request, 'ml_models/notes.html')   