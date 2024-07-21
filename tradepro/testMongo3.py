from pymongo import MongoClient
from bson import ObjectId  # Required to work with ObjectId in pymongo

def retrieve_documents(collection_name, db_name='tradepro_dev'):
    """
    Function to retrieve all documents from a MongoDB collection.

    Args:
        collection_name (str): The name of the MongoDB collection.
        db_name (str): The name of the database. Default is 'tradepro_dev'.

    Returns:
        dict: The retrieved documents or an error message if the collection is empty.
    """
    # MongoDB connection string (replace with your connection string)
    MONGO_URI = "mongodb+srv://Sunny:Xjkdxjkd1@cluster0.ars0ie4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    # Create a new client and connect to the server
    client = MongoClient(MONGO_URI)
    
    # Access the database and collection
    db = client[db_name]  # Database name
    collection = db[collection_name]  # Collection name

    try:
        # Retrieve all documents from the collection
        documents = list(collection.find({}))

        # Check if any documents were found and return the result
        if documents:
            return {"status": "success", "documents": documents}
        else:
            return {"status": "error", "message": f"No documents found in collection: {collection_name}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Example usage
    collection_name = 'USDJPY_pred_continue_1h_v5'
    result = retrieve_documents(collection_name)

    # Print the result
    if result['status'] == 'success':
        print(f"Documents retrieved successfully from collection '{collection_name}':")
        for doc in result['documents']:
            print(doc)
    else:
        print(f"Error: {result['message']}")
