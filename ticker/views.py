from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def tickers(request):
    page = 'Tickers'
    return render(request, 'ticker/tickers.html', {'page':page})


def ticker(request, pk):
    tickerObj = None
 
    return render(request, 'ticker/ticker.html')

