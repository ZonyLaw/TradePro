
from pymongo.mongo_client import MongoClient

# uri = "mongodb+srv://Sunny:yogayoga1@cluster0.ars0ie4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
uri = "mongodb+srv://slaw1:Xjkdxjkd1@cluster0.ars0ie4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)