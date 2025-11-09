from typing import List, Optional
from bson import ObjectId  # Add this import
from datetime import datetime, timezone, timedelta

# Helper function to serialize MongoDB documents
def serialize_todo(todo):
    """Convert MongoDB document to JSON-serializable dict"""
    if todo is None:
        return None
    
    serialized = {}
    for key, value in todo.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized
