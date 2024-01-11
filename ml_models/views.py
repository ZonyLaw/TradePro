from django.shortcuts import render, get_object_or_404
from .utils.predictive_analysis import standard_analysis, model_run
from .utils.manual_model_input import manual_price_input
from .form import ModelParameters
from prices.models import Price
from tickers.models import Ticker
import pandas as pd
from .utils.utils import trade_direction


# Create your views here.
def ml_predictions(request):
    """
    #outstanding...data should be saved as a text and accessed as static to avoid running the code again.
    This is a view function that pass the model predictions to the the frontend.
    Predictions is saved as dictionary of array containing the values of each profit/loss cateogires.
    """
    # predictions={'fist':['12','23'],'second':['12','23']}
    
    pred_reverse, pred_continue, pred_historical = standard_analysis()
    print(pred_historical)
    
    ticker_instance = get_object_or_404(Ticker, symbol="USDJPY")
    prices = Price.objects.filter(ticker=ticker_instance)
    
    #sort prices table in descending to get the latest price on top
    sorted_prices = prices.order_by('-date')

    #get the latest price    
    last_price = sorted_prices[0]

    #get the price 4 hr ago but not sure how useful it will be yet
    last_four_prices = sorted_prices[:5]
    prices_df = pd.DataFrame(list(last_four_prices.values()))

    # Access prices using index, note that len() is 'index + 1'
    if len(prices_df) >= 5:
        fourth_last_price = prices_df.loc[4]
        print(prices_df)
    else:
        fourth_last_price = None
    
    trade_diff_1hr = last_price.close - last_price.open
    trade_diff_4hr = last_price.close - fourth_last_price.open
    
    trade = {"one" :trade_direction(trade_diff_1hr),
    "four": trade_direction(trade_diff_4hr)}
    
    candle_size = {"one" :trade_diff_1hr,
    "four": trade_diff_4hr}
      
    context={'pred_reverse': pred_reverse, 'pred_continue':pred_continue, 'pred_historical': pred_historical, 'trade':trade, 'candle_size': candle_size, 'last_price':last_price, 'fourth_last_price':fourth_last_price}
    
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

    