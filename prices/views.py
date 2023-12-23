from django.shortcuts import render, redirect,  get_object_or_404, HttpResponse
from django.urls import reverse 
from tickers.models import Ticker
from prices.models import Price
from .form import PriceForm

from .form import FileUploadForm
from .utils import import_prices_from_csv, generate_csv

from django.utils import timezone
from datetime import datetime


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

    return render(request, 'prices/price_form.html', {'form': form})


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

    return render(request, 'prices/price_form.html', {'form': form})


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
        form = FileUploadForm()
        
        return render(request, 'prices/export_prices.html', {'tickers': tickers_db})


def delete_price(request, pk):
    price = get_object_or_404(Price, id=pk)
    ticker_id = price.ticker.id
  
    if request.method == 'POST':
        price.delete()
        return redirect('ticker', ticker_id)  # Redirect to the page displaying the list of prices

    context = {'object': price, 'ticker_id': ticker_id}
   
    return render(request, 'prices/price_delete_template.html', context)


def delete_prices(request):
    # Get the 'from' and 'to' date parameters from the request
    from_date_str = request.GET.get('from_date', '')
    to_date_str = request.GET.get('to_date', '')

    try:
        # Convert the date strings to datetime objects
        from_date = datetime.strptime(from_date_str, "%Y-%m-%d")
        to_date = datetime.strptime(to_date_str, "%Y-%m-%d")

        # Assuming the model has a 'timestamp' field
        # Delete prices within the specified date range
        Price.objects.filter(timestamp__range=(from_date, to_date)).delete()

        return HttpResponse("Prices deleted successfully within the specified date range.")
    except ValueError:
        # Handle invalid date format
        return HttpResponse("Invalid date format. Please use YYYY-MM-DD.")
