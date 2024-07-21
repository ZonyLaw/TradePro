from pymongo import MongoClient

uri = "mongodb://Sunny:yogayoga1@cluster0-shard-00-00.ars0ie4.mongodb.net:27017/?ssl=false"
client = MongoClient(uri)

try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error: {e}")
