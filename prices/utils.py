import csv
import os
from io import StringIO
from django.shortcuts import HttpResponse
from datetime import datetime
from .models import Price
from tickers.models import Ticker


def import_prices_from_csv(uploaded_file):
    """
    This function is to upload .csv file of historical prices.
    The contents expected are: date, ticker, open, close, high, & low prices.
    It allows you to store the uploaded file temporarily on the server, 
    process it, and then remove the temporary file. 

    Args:
        uploaded_file (object): this is extracted from request.FILES that the user specified

    Returns:
        _type_: returns a message that upload is successful
    """
    
    # Handle the file in-memory or save it to a specific location
    # For example, you can save it to a temporary location
    #update needed!!
    try:
        with open('prices/data/temp_file.csv', 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Now, you have the filepath (though temporary) to process
        file_path = 'prices/data/temp_file.csv'

        # Process the file as you were doing before
        with open(file_path, 'r') as csvfile:
            data = csv.DictReader(csvfile)
            for row in data:
                ticker_obj = Ticker.objects.get(symbol=row["ticker"])

                # Reformatting date for DB
                date_str = row['date']
                date_obj = datetime.strptime(date_str, "%d/%m/%Y %H:%M")

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
    except:
        print("error")

    return HttpResponse("CSV import completed.")


def generate_csv(prices):
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)

    # Write header
    csv_writer.writerow(['date', 'ticker', 'open', 'close', 'high', 'low'])

    # Write data
    for price in prices:
        csv_writer.writerow([price.date, price.ticker, price.open, price.close, price.high, price.low])

    # Get CSV data as a string
    csv_data = csv_buffer.getvalue()

    # Close the buffer
    csv_buffer.close()

    return csv_data