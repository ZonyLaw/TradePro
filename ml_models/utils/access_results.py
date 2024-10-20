import os
import json
import csv
from django.conf import settings
from pymongo import MongoClient

from pymongo.errors import OperationFailure
from datetime import datetime
from bson import ObjectId

# Access the MongoDB URI from settings
MONGO_URI = settings.DATABASES['mongo']['CLIENT']['host']

# Initialize MongoClient with the URI from settings
client = MongoClient(MONGO_URI)

# Access the database
db = client[settings.DATABASES['mongo']['NAME']]


def write_to_json(data, model_ticker, filename):
    """
    This function writes the data to a json file as a temporary storage.

    Args:
        data (dictionary): containing the model prediction results
        filename (string): filename of the json file.
    """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Assuming this is in a module

    # Move up two levels from the current module's directory
    base_dir_up_two_levels = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))

    relative_path = os.path.join(base_dir_up_two_levels, 'media', 'model_results', model_ticker, filename)
    # Construct the absolute path
    absolute_path = os.path.join(base_dir, relative_path)

    # Ensure that the directory structure exists, creating directories if necessary
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    # Write to the JSON file
    with open(absolute_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def read_prediction_from_json(model_ticker, filename):
    """_summary_

    Args:
        filename (string): the json filename where the model resutls is saved.

    Returns:
        dictionary: the data contained in the json file.
    """
    
    levels_to_move_up = 2
    
    current_file_path = os.path.dirname(os.path.abspath(__file__))  # Assuming this is in a module

    # Move up the specified number of levels
    for _ in range(levels_to_move_up):
        current_file_path = os.path.dirname(current_file_path)

    relative_path = os.path.join( 'media','model_results', model_ticker, filename)
    # Construct the absolute path
    absolute_path = os.path.join(current_file_path, relative_path)
    print("absolute_path>>>>>",absolute_path)
    
    with open(absolute_path, 'r') as file:
        data = json.load(file)
    
    return data


def write_to_csv(comment1, comment2, filename):
    """
    This function appends the comments to a CSV file or creates a new file if it doesn't exist.

    Args:
        comment1 (string): comment on the model's results
        comment2 (string): another comment on the model's results
        filename (string): filename of the CSV file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Assuming this is in a module

    # Move up two levels from the current module's directory
    base_dir_up_two_levels = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))

    relative_path = os.path.join(base_dir_up_two_levels, 'media', 'model_results', filename)
    # Construct the absolute path
    absolute_path = os.path.join(base_dir, relative_path)

    # Ensure that the directory structure exists, creating directories if necessary
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    # Check if file exists
    file_exists = os.path.isfile(absolute_path)

    # Write or append to the CSV file
    with open(absolute_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # If file doesn't exist, write header
        if not file_exists:
            writer.writerow(["Comment 1", "Comment 2"])
        # Write comments as rows
        writer.writerow([comment1, comment2])


def write_to_mongo(collection_name, data):
    """
    Function to delete an existing collection and insert a new document into a MongoDB collection.

    Args:
        collection_name (str): The name of the MongoDB collection.
        data (dict): The document to be inserted into the collection.
    """
    try:
        # Check if the collection exists
        if collection_name in db.list_collection_names():
            # Drop the existing collection
            db.drop_collection(collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")

        # Access the collection
        collection = db[collection_name]

        # Insert the new document
        collection.insert_one(data)
        print("Insert operation completed.")
        # print(f"New document inserted with ID: {result.inserted_id}")

    except OperationFailure as e:
        print(f"Operation failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        

def write_append_to_mongo(collection_name, data):
    """
    Function to delete an existing collection and insert a new document into a MongoDB collection.

    Args:
        collection_name (str): The name of the MongoDB collection.
        data (dict): The document to be inserted into the collection.
    """
    try:


        # Access the collection
        collection = db[collection_name]

        # Insert the new document
        collection.insert_one(data)
        print("Insert operation completed.")
        # print(f"New document inserted with ID: {result.inserted_id}")

    except OperationFailure as e:
        print(f"Operation failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_prediction_from_Mongo(collection_name):
    """
    Reads prediction data from MongoDB for a given collection.

    Args:
        collection_name (string): The name of the collection in the database.

    Returns:
        list: A list of dictionaries containing the data from the collection.
    """

    try:
        # Access the collection from the global db
        collection = db[collection_name]

        documents = list(collection.find())
 
        # Remove MongoDB specific '_id' field if it's not needed
        if documents:
            document = documents[0]  # Get the first document in the array
            document.pop('_id', None)  # Remove '_id' field if present
            return document
        
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def read_list_from_Mongo(collection_name):
    try:
        # Access the collection from the global db
        collection = db[collection_name]

        documents = list(collection.find())
 
        
        return documents
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    

