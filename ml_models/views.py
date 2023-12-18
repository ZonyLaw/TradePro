from django.shortcuts import render
from .utils.predictive_analysis import model_run

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

