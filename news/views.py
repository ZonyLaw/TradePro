from django.shortcuts import render
from .form import FileUploadForm, CurrencyFilterForm 
from django.conf import settings
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from django.core.files.storage import FileSystemStorage
from collections import defaultdict
import csv
import datetime
import pandas as pd


# Create your views here.
# Access the MongoDB URI from settings
MONGO_URI = settings.DATABASES['mongo']['CLIENT']['host']

# Initialize MongoClient with the URI from settings
client = MongoClient(MONGO_URI)

# Access the database
db = client[settings.DATABASES['mongo']['NAME']]

def upload_news(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']

            # Use pandas to read the uploaded file directly into a DataFrame
            df = pd.read_csv(file)

            # Drop the existing collection to start fresh
            collection = db['news_collection']
            collection.drop()

            # Create indexes for faster retrieval
            collection.create_index([('currency', 1), ('date', 1), ('time', 1)])

            # Create a nested structure to hold news items by currency and date
            data = defaultdict(lambda: defaultdict(list))

            for index, row in df.iterrows():
                news_item = {
                    'impact': row['impact'],
                    'event': row['event'],
                    'actual': row['actual'],
                    'forecast': row['forecast'],
                    'previous': row['previous'],
                    'outcome': row['outcome'],
                    'time': row['time']  # Keep time if needed for future reference
                }
                # Append news item to the corresponding currency and date
                data[row['currency']][row['date']].append(news_item)
            
            # Prepare the final document structure
            final_document = {
                'data': data
            }

            # Insert the combined data as one document in MongoDB
            collection.insert_one(final_document)

            return render(request, 'news/upload_success.html')
    else:
        form = FileUploadForm()
    return render(request, 'news/upload_news.html', {'form': form})


def get_current_week_date_range():
    today = datetime.date.today()
    start_of_week = today - datetime.timedelta(days=today.weekday())  # Monday
    end_of_week = start_of_week + datetime.timedelta(days=6)  # Sunday
    return start_of_week, end_of_week


def get_news(request):
    news_data = {}
    form = CurrencyFilterForm(request.GET or None)

    if form.is_valid():
        currency = form.cleaned_data['currency']
        start_of_week, end_of_week = get_current_week_date_range()
        collection = db['news_collection']
        
        try:
            result = collection.find_one({
                'data': {
                    '$exists': True
                }
            })
           
            if result:
                # Convert date strings to date objects
                parsed_news_data = {}
                for date_str, news_list in result['data'][currency].items():
                    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                    if start_of_week <= date_obj <= end_of_week:
                        parsed_news_data[date_obj] = news_list
                
                news_data = parsed_news_data
        except OperationFailure as e:
            print(f"Error retrieving data from MongoDB: {e}")
            
    return render(request, 'news/get_news.html', {'news_data': news_data, 'form': form})
