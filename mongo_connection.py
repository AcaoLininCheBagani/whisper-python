# server.py
import os

# MongoDB & Environment
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId  # Add this import

load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in environment")

client = AsyncIOMotorClient(MONGO_URI)
db = client.track_my_todo

