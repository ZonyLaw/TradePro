from django.shortcuts import render
from .form import FileUploadForm
from django.conf import settings
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from django.core.files.storage import FileSystemStorage
from collections import defaultdict
import csv

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
            print("path below")
            print(file_path)
            collection = db['news_collection']
            
 # Create indexes for faster retrieval
            collection.create_index([('currency', 1), ('date', 1), ('time', 1)])
            
            # Read and process the CSV file
            with open(file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Print out the reader content for debugging
                for row in reader:
                    print(row)  # Print each row as a dictionary
                
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