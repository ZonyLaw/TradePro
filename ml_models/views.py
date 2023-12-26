from django.shortcuts import render, redirect
from .utils.predictive_analysis import standard_analysis, model_run
from .utils.manual_model_input import manual_price_input
from .form import ModelParameters


# Create your views here.
def ml_predictions(request):
    """
    #outstanding...data should be saved as a text and accessed as static to avoid running the code again.
    This is a view function that pass the model predictions to the the frontend.
    Predictions is saved as dictionary of array containing the values of each profit/loss cateogires.
    """
    # predictions={'fist':['12','23'],'second':['12','23']}
    
    pred_reverse, pred_continue = standard_analysis()
    context={'pred_reverse': pred_reverse, 'pred_continue':pred_continue}
    
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

    