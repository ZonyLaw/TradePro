from django.shortcuts import render
from .utils.predictive_analysis import model_run

# Create your views here.
def ml_predictions(request):
    #this will recieve a dictionary of the results
    predictions = model_run
    context={'predictions': predictions}
    
    
    return render(request, 'ml_model/ml_predictions.html', context)

