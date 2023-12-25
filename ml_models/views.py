from django.shortcuts import render
from .utils.predictive_analysis import model_run
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
            print("to be save")

    else:
        # Initialize the form with default values
        form = ModelParameters(initial={
            'ma20': 0.0,
            'ma50': 0.0,
            'ma100': 0.0,
            'bb_high': 0.0,
            'bb_low': 0.0,
            'open_price': 0.0,
            'close_price': 0.0,
            'high_price': 0.0,
            'low_price': 0.0,
        })

    
    context = {'form': form}
    
    return render(request, 'ml_models/ml_parameters_form.html', context)