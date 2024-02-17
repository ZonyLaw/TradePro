from django.shortcuts import render, redirect,  get_object_or_404, HttpResponse
from tickers.models import Ticker
from prices.models import Price
from .form import PriceForm, PriceRangeForm

from .form import FileUploadForm
from .utils import import_prices_from_csv, generate_csv

from datetime import datetime
from django.conf import settings
from prices.IG_session.run_IG import run_IG
from ml_models.utils.predictive_analysis import run_model_predictions


def createPrice(request):
    
    page = "Enter new price"
    
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
        
    context = {"page":page, 'form': form}

    return render(request, 'prices/price_form.html', context)


def updatePrice(request, pk):
    
    page = "Update Price"
    # Get the existing Price instance
    price_instance = get_object_or_404(Price, id=pk)

    if request.method == 'POST':
        # Pass the existing instance to the form
        form = PriceForm(request.POST, instance=price_instance)

        if form.is_valid():
            # Save the updated price
            form.save()
            run_model_predictions()

            # Redirect to the appropriate page
            return redirect('tickers')  # Replace 'tickers' with your actual URL name
    else:
        # Populate the form with the existing data
        form = PriceForm(instance=price_instance)
    
    context = {"page":page, 'form': form}

    return render(request, 'prices/price_form.html', context)



def upload_prices(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            print("File is valid!", file)
            import_prices_from_csv(file)
            return redirect('tickers')
        else:
            print("Form is not valid:", form.errors)
    else:
        form = FileUploadForm()

    return render(request, 'prices/upload_prices.html', {'form': form})


def export_prices(request):
    tickers_db = Ticker.objects.all()
    if request.GET:
        ticker = request.GET.get('ticker', None)

        if ticker:
            ticker_instance = get_object_or_404(Ticker, symbol=ticker)
            prices = Price.objects.filter(ticker=ticker_instance)
            csv_data = generate_csv(prices)
            filename = f"{ticker}_prices.csv"
        else:
            prices = Price.objects.all()
            csv_data = generate_csv(prices)
            filename = "all_prices.csv"

        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        return response

    else:
        
        return render(request, 'prices/export_prices.html', {'tickers': tickers_db})


def delete_price(request, pk):
    price = get_object_or_404(Price, id=pk)
    ticker_id = price.ticker.id
  
    if request.method == 'POST':
        price.delete()
        return redirect('ticker', ticker_id)  # Redirect to the page displaying the list of prices

    context = {'object': price, 'ticker_id': ticker_id}
   
    return render(request, 'prices/price_delete_template.html', context)


def delete_prices_range(request):
    if request.method == 'POST':
        form = PriceRangeForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            ticker_instance = Ticker.objects.get(symbol=ticker)
           
            # Delete prices within the specified date range
            Price.objects.filter(ticker=ticker_instance, date__range=[start_date, end_date]).delete()
            
            return redirect('tickers')  # Redirect to a success page or another view
    else:
        form = PriceRangeForm()

    return render(request, 'prices/prices_range.html', {'form': form})

def get_IG_prices(request):
        
    if request.method == 'POST':
        form = PriceRangeForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            ticker_instance = Ticker.objects.get(symbol=ticker)
            ticker_symbol = ticker_instance.symbol
            
            print("ticker provide in view>>>", ticker)
            run_IG(ticker_symbol, start_date, end_date)
            return redirect('tickers')  # Redirect to a success page or another view
    else:
        form = PriceRangeForm()

    return render(request, 'prices/prices_range.html', {'form': form})
