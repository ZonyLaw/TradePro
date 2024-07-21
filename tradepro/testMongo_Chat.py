
from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi

# uri = "mongodb+srv://Sunny:Xjkdxjkd1@cluster0.ars0ie4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
uri = "mongodb+srv://Sunny:Xjkdxjkd1@cluster0.ars0ie4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tlsCAFile=isrgrootx1.pem"




# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)