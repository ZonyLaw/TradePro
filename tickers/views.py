from django.shortcuts import render, redirect
from .models import Ticker
from prices.models import Price
from .form import TickerForm

# Create your views here.
def tickers(request):
    tickers = Ticker.objects.all()
    context = {'tickers': tickers}
    return render(request, 'tickers/tickers.html', context)


def ticker(request, pk):
    ticker = Ticker.objects.get(id=pk)
    prices = ticker.price_set.all()
    tickerObj = None
    context = {'ticker': ticker, 'prices': prices}
    return render(request, 'tickers/ticker.html', context)


def createTicker(request):
    form = TickerForm()
    
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('tickers')
    
    context = {'form': form}
    return render(request, "tickers/ticker_form.html", context)

def updateTicker(request, pk):
    ticker = Ticker.objects.get(id=pk)
    form = TickerForm(instance=ticker)
    
    if request.method == 'POST':
        form = TickerForm(request.POST, instance=ticker)
        if form.is_valid():
            form.save()
            return redirect('tickers')
    
    context = {'form': form}
    return render(request, "tickers/ticker_form.html", context)

def deleteTicker(request, pk):
    ticker = Ticker.objects.get(id=pk)
    if request.method == 'POST':
        ticker.delete()
        return redirect('tickers')
    
    context = {'object': ticker}
    print(ticker)
    return render(request, 'tickers/delete_template.html', context)

