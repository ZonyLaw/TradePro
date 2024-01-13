from django.shortcuts import render, get_object_or_404, HttpResponse
from .utils.predictive_analysis import standard_analysis, model_run, trade_forecast_assessment
from .utils.manual_model_input import manual_price_input
from .form import ModelParameters
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
    
    pred_reverse, pred_continue, pred_historical = standard_analysis()
    
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
    
    context={'pred_reverse': pred_reverse, 'pred_continue':pred_continue, 'pred_historical': pred_historical, 
             'open_prices': open_prices, 'close_prices': close_prices,'trade':trade, 'candle_size':candle_size}
    
    return render(request, 'ml_models/ml_predictions.html', context)


def ml_manual(request):
    
    """
    This is a function to get user input for running the model manually.
    The results are displayed in the same webpage as the input form.
    User can update input to get new results.
    
    """
    
    form = ModelParameters(request.POST)
    
    if request.method == 'POST':
        form = ModelParameters(request.POST)
        if form.is_valid():
            model_input = manual_price_input(form)
            results = model_run(model_input)
            request.session['ml_results'] = results
            
    else:
        
        results = ''
        # Initialize the form with default values
        form = ModelParameters(initial={
            'open': 0.0,
            'close': 0.0,
            'open_lag1': 0.0,
            'close_lag1': 0.0,
            'ma50': 0.0,
            'close_4': 0.0,
            'ma20_4': 0.0,
            'ma50_4': 0.0,
            'ma100_4': 0.0,
            'bb20_high': 0.0,
            'bb20_low': 0.0,
            'bb20_high_4': 0.0,
            'bb20_low_4': 0.0,
            'hour': 0,
            'trend_strength_1':0,
            'bb_status_1': "inside_bb",

        })

    context = {'form':form, 'results':results}
    
    return render(request, 'ml_models/ml_manual_analysis.html', context)

def export_model_results(request):
    
    if request.method == 'POST':
    
        model_results = trade_forecast_assessment()
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="model_results.csv"'
        model_results.to_csv(path_or_buf=response, index=False, encoding='utf-8')

        return response
    
    else:
        return render(request, 'ml_models/export_model_results.html')
    