from django.shortcuts import render, redirect
from .models import Ticker
from .form import TickerForm

# Create your views here.
def tickers(request):
    tickers = Ticker.objects.all()
    context = {'tickers': tickers}
    return render(request, 'ticker/tickers.html', context)


def ticker(request, pk):
    ticker = Ticker.objects.get(id=pk)
    tickerObj = None
    context = {'ticker': ticker}
    return render(request, 'ticker/ticker.html', context)


def createTicker(request):
    form = TickerForm()
    
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('tickers')
    
    context = {'form': form}
    return render(request, "ticker/ticker_form.html", context)

def updateTicker(request, pk):
    ticker = Ticker.objects.get(id=pk)
    form = TickerForm(instance=ticker)
    
    if request.method == 'POST':
        form = TickerForm(request.POST, instance=ticker)
        if form.is_valid():
            form.save()
            return redirect('tickers')
    
    context = {'form': form}
    return render(request, "ticker/ticker_form.html", context)

def deleteTicker(request, pk):
    ticker = Ticker.objects.get(id=pk)
    if request.method == 'POST':
        ticker.delete()
        return redirect('tickers')
    
    context = {'object': ticker}
    print(ticker)
    return render(request, 'ticker/delete_template.html', context)