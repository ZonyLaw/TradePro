from django.shortcuts import render, redirect,  get_object_or_404
from ticker.models import Ticker
from price.models import Price
from .form import PriceForm


def createPrice(request):
    if request.method == 'POST':
        
        form = PriceForm(request.POST)
        # print(form)
        
        if form.is_valid():
            # Save the price, associating it with the specified ticker
            ticker_instance = form.cleaned_data['ticker']

            # Create the Price instance and assign the Ticker instance
            price_instance = form.save(commit=False)
            price_instance.ticker = ticker_instance
            price_instance.save()

            # messages.success(request, 'Price created successfully.')
            return redirect('tickers')  # Redirect to the home page or any other page
    else:
        form = PriceForm()

    return render(request, 'price/price_form.html', {'form': form})


def updatePrice(request, pk):
    
    price_instance = get_object_or_404(Price, id=pk)
        
    if request.method == 'POST':
        
        form = PriceForm(request.POST)
        # print(form)
        
        if form.is_valid():
            # Save the price, associating it with the specified ticker
            ticker_instance = form.cleaned_data['ticker']

            # Create the Price instance and assign the Ticker instance
            price_instance = form.save(commit=False)
            price_instance.ticker = ticker_instance
            price_instance.save()

            # messages.success(request, 'Price created successfully.')
            return redirect('tickers')  # Redirect to the home page or any other page
    else:
        form = PriceForm(instance=price_instance)

    return render(request, 'price/price_form.html', {'form': form})