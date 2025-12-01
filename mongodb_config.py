import pymongo
import os
from dotenv import load_dotenv

# Database
load_dotenv()
mongo_url = os.getenv("MONGO_URL")
client = pymongo.MongoClient(mongo_url)
db = client["OCR_Database"]