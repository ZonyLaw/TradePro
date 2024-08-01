from django.shortcuts import render
from .form import FileUploadForm, CurrencyFilterForm 
from django.conf import settings
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from django.core.files.storage import FileSystemStorage
from collections import defaultdict
import csv
import datetime


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
            print(file)
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)
            collection = db['news_collection']
            
 # Create indexes for faster retrieval
            collection.create_index([('currency', 1), ('date', 1), ('time', 1)])
            
            # Read and process the CSV file
            with open(file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Rewind the reader to the beginning of the file
                csvfile.seek(0)
                reader = csv.DictReader(csvfile)

                # Create a nested structure to hold news items by currency and date
                data = defaultdict(lambda: defaultdict(list))

                for row in reader:
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
            
            return render(request, 'news/upload_success.html', {'file_url': fs.url(filename)})
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
                news_data = {
                    date: news_list
                    for date, news_list in result['data'][currency].items()
                    if start_of_week <= datetime.datetime.strptime(date, '%Y-%m-%d').date() <= end_of_week
                }
        except OperationFailure as e:
            print(f"Error retrieving data from MongoDB: {e}")
            
    return render(request, 'news/get_news.html', {'news_data': news_data, 'form': form})