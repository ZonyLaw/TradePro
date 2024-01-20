from django.shortcuts import render, get_object_or_404, HttpResponse
from .utils.predictive_analysis import standard_analysis, model_run, trade_forecast_assessment
from .utils.data_processing import stats_df_gen
from .utils.manual_model_input import manual_price_input
from .form import ModelParameters, ModelSelection
from prices.models import Price
from tickers.models import Ticker
import pandas as pd
from .utils.utils import trade_direction


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
        
        
    pred_reverse, pred_continue, pred_historical = standard_analysis(model_version)
    
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
    trade_diff_1hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[3]['open']
    trade_diff_4hr = last_four_prices_df.iloc[3]['close'] - last_four_prices_df.iloc[0]['open']
    
    trade = {"one" :trade_direction(trade_diff_1hr),
    "four": trade_direction(trade_diff_4hr)}
    
    candle_size = {"one" :trade_diff_1hr,
    "four": trade_diff_4hr}
    
    context={'form': form, 'pred_reverse': pred_reverse, 'pred_continue':pred_continue, 'pred_historical': pred_historical, 
             'open_prices': open_prices, 'close_prices': close_prices,'trade':trade, 'candle_size':candle_size}
    
    return render(request, 'ml_models/ml_predictions.html', context)


def ml_manual(request):
    
    """
    This is a function to get user input for running the model manually.
    The results are displayed in the same webpage as the input form.
    User can update input to get new results.
    NOTE: the results are slightly different becuase the 4hr close is assumed to be same as 1hr close.
    
    """
    
    form = ModelParameters(request.POST)
    
    ticker_instance = get_object_or_404(Ticker, symbol="USDJPY")
    prices = Price.objects.filter(ticker=ticker_instance)
    
    prices_df = pd.DataFrame(list(prices.values()))
    prices_df = prices_df.sort_values(by='date', ascending=True)
    last_prices_df = prices_df.tail(2)
    last_price_stats = stats_df_gen(prices_df,2)
    
    results = []
    if request.method == 'POST':
        form = ModelParameters(request.POST)
        if form.is_valid():
            user_input = manual_price_input(form)
            results, _, _, _, _ = model_run(user_input)            
     
    else:
           
        # Initialize the form with default values
        form = ModelParameters(initial={           
            
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
    