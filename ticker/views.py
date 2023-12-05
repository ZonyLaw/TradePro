from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def tickers(request):
    return render(request, 'ticker/tickers.html')


def ticker(request, pk):
    return render(request, 'ticker/ticker.html')

