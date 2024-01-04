from django.shortcuts import render, redirect
from .models import Ticker
from .form import TickerForm
from users.models import Profile

from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required

# Create your views here.
def tickers(request):
    tickers = Ticker.objects.all()
    context = {'tickers': tickers}
    return render(request, 'tickers/tickers.html', context)


@login_required
def ticker(request, pk):
    ticker = get_object_or_404(Ticker, id=pk)
    prices = ticker.price_set.all()
    
    profile = Profile.objects.get(user = request.user)
    user_timezone = profile.timezone
    # print(user_timezone)
    # Fetch prices in UTC for the specified ticker

    # Convert prices to user's timezone
    prices_user_timezone = []
    for price in prices:
        localized_date = price.date.astimezone(user_timezone)
        prices_user_timezone.append({'id':price.id, 'date': localized_date, 
                                     'open': price.open , 'close': price.close,
                                     'trade': 'Buy' if price.open - price.close < 0 else 'Sell'})
        sorted_prices = sorted(prices_user_timezone, key=lambda x: x['date'], reverse=True)
    
    context = {'ticker': ticker, 'prices': sorted_prices,
               'user_timezone':user_timezone}
    return render(request, 'tickers/ticker.html', context)

#this is a simple version of showing the date time but will use setting timezone
def ticker2(request, pk):
    profile = Profile.objects.get(user = request.user)
    print("test", profile.timezone)
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

