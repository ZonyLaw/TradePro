import os
from pymongo import MongoClient
from pymongo.monitoring import ServerHeartbeatListener
from importlib.metadata import version
from pymongo.driver_info import DriverInfo

MONGO_URI= "mongodb+srv://Sunny:Xjkdxjkd1@cluster0.ars0ie4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&tlsCAFile=mongodb.pem"

# Function to get package version
def get_package_version(package_name):
    try:
        return version(package_name)
    except Exception as e:
        return "unknown"

# Your custom driver name and version
driver_name = "llama-index"
driver_version = get_package_version(driver_name)

# Environment variable for MongoDB URI
mongo_uri = MONGO_URI
if not mongo_uri:
    raise EnvironmentError("MONGO_URI environment variable not set")

# Path to your PEM file for TLS/SSL
tls_certificate_key_file = r'C:\Users\sunny\Desktop\Development\python\TradePro\private_key.pem'

# Create the MongoClient with TLS/SSL configuration
mongodb_client = MongoClient(
    mongo_uri,
    driver=DriverInfo(name=driver_name, version=driver_version),
    tls=True,
    tlsCertificateKeyFile=tls_certificate_key_file,
    tlsAllowInvalidCertificates=True
)

# Example usage: listing database names
try:
    db_list = mongodb_client.list_database_names()
    print("Databases:", db_list)
except Exception as e:
    print("An error occurred:", e)
