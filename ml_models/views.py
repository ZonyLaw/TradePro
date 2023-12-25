from django.shortcuts import render, redirect
from .utils.predictive_analysis import model_run
from .utils.manual_model_input import manaul_price_input
from .form import ModelParameters


# Create your views here.
def ml_predictions(request):
    """
    #outstanding...data should be saved as a text and accessed as static to avoid running the code again.
    This is a view function that pass the model predictions to the the frontend.
    Predictions is saved as dictionary of array for the values of each profit/loss cateogires.
    """
    # predictions={'fist':['12','23'],'second':['12','23']}
    pred_reverse, pred_continue = model_run()
    context={'pred_reverse': pred_reverse, 'pred_continue':pred_continue}
    
    return render(request, 'ml_models/ml_predictions.html', context)


def ml_parameters(request):
    
    form = ModelParameters(request.POST)
    
    if request.method == 'POST':
        form = ModelParameters(request.POST)
        if form.is_valid():
            model_input = manaul_price_input(form)
            results = "hi there from another function"
            print("to be save")
            request.session['ml_results'] = results
            
            return redirect('ml-manual-run')

    else:
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

    
    context = {'form': form}
    
    return render(request, 'ml_models/ml_parameters_form.html', context)


def ml_manual_run(request):
    
     # Retrieve the results from the session
    results = request.session.get('ml_results', None)

    # Check if results are available in the session
    if results is None:
        return render(request, 'ml_models/ml_manual_run.html', {'error_message': 'Results not found. Please input data first.'})

    # Use 'results' in your template or view logic
    return render(request, 'ml_models/ml_manual_run.html', {'results': results})
    
    
    