import csv
import os
from django.shortcuts import HttpResponse
from datetime import datetime
from .models import Price
from .models import Ticker


def import_prices_from_csv(uploaded_file):
    """
    This function is to upload .csv file of historical prices.
    The contents expected are: date, tikcer, open, close, high, & low prices.

    Args:
        uploaded_file (object): this is extracted from request.FILES that the user specified

    Returns:
        _type_: returns a message that upload is successful
    """
    
    # Handle the file in-memory or save it to a specific location
    # For example, you can save it to a temporary location
    with open('price/data/temp_file.csv', 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    # Now, you have the file path (though temporary) to process
    file_path = 'price/data/temp_file.csv'

    # Process the file as you were doing before
    with open(file_path, 'r') as csvfile:
        data = csv.DictReader(csvfile)
        for row in data:
            ticker_obj = Ticker.objects.get(symbol=row["ticker"])

            # Reformatting date for DB
            date_str = row['date']
            date_obj = datetime.strptime(date_str, "%m/%d/%Y %H:%M")

            open_price = float(row['open'])
            close_price = float(row['close'])
            high_price = float(row['high'])
            low_price = float(row['low'])

            # Create or update the Price object
            price, created = Price.objects.update_or_create(
                ticker=ticker_obj,
                date=date_obj,
                defaults={'open': open_price, 'close': close_price, 'high': high_price, 'low': low_price}
            )

            print(f"Price {'created' if created else 'updated'} for date {date_obj}")

    os.remove(file_path)

    return HttpResponse("CSV import completed.")


def import_prices_from_csv2(request):
    """
    This function is to upload .csv file of historical prices.
    The contents expected are: date, tikcer, open, close, high, & low prices.
    This was initial setup for testing and could be deleted.

    Args:
        request (object): this is not used in the function

    Returns:
        _type_: returns a message that upload is successful
    """

    file_path = r'C:\Users\sunny\Desktop\Development\python\TradePro\USDJPY_prices.csv'
    print(file_path)
    with open(file_path, 'r') as csvfile:
        data = csv.DictReader(csvfile)
        for row in data:
            ticker_obj = Ticker.objects.get(symbol=row["ticker"])
            
            # reformatting date for DB
            date_str = row['date']
            date_obj = datetime.strptime(date_str, "%m/%d/%Y %H:%M")

            # Assuming 'close' is a numeric field
            open_price = float(row['open'])
            close_price = float(row['close'])
            high_price = float(row['high'])
            low_price = float(row['low'])

            # Create or update the Price object
            
            price, created = Price.objects.update_or_create(
                ticker=ticker_obj,
                date=date_obj,
                defaults={'open': open_price, 'close': close_price, 'high':high_price, 'low':low_price}
            )

            print(f"Price {'created' if created else 'updated'} for date {date_obj}")
            
    return HttpResponse("CSV import completed.")

# Example usage
# file_path = r'C:\Users\sunny\Desktop\Development\python\TradePro\USDJPY_prices.csv'
# import_prices_from_csv(file_path)
