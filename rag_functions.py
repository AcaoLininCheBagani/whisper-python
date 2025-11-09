from typing import List, Optional
from mongo_connection import db
from helper import serialize_todo
from datetime import datetime, timezone, timedelta

def build_context_from_todos(todos: List[dict]) -> str:
    """Build a readable context string from similar todos"""
    if not todos:
        return "No relevant todo items found."
    
    context_parts = []
    for i, todo in enumerate(todos, 1):
        status = "✓ Completed" if todo.get('completed') else "○ Pending"
        context_parts.append(
            f"{i}. {todo['title']} (Status: {status})"
        )
    return "\n".join(context_parts)


# Add this function to handle date-based queries
def is_date_query(text: str) -> bool:
    """Check if the query is about dates/today"""
    date_keywords = ['today', 'created today', 'made today', 'this day', 'daily']
    return any(keyword in text.lower() for keyword in date_keywords)

async def get_todos_created_today():
    """Get todos created today using regular MongoDB query"""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    
    todos = await db.todos.find({
        "createdAt": {
            "$gte": today_start,
            "$lt": today_end
        }
    },{"title": 1, "completed": 1}).to_list(length=100)
    
    return [serialize_todo(todo) for todo in todos]

def is_yesterday(text: str) -> bool:
    date_keywords = ['yesterday', 'last week', 'this week']
    return any(keyword in text.lower() for keyword in date_keywords)

async def get_todos_created_yesterday():
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    yesterday_end = today_start
    
    todos = await db.todos.find({
        "createdAt": {
            "$gte": yesterday_start,
            "$lt": yesterday_end
        }
    },{"title": 1, "completed": 1}).to_list(length=100)
    
    return [serialize_todo(todo) for todo in todos]

def get_user_intent(text: str) -> str:
    user_intents = ['add', 'update', 'delete']
    
    matched_keyword = None
    for keyword in user_intents:
        if keyword in text.lower().split():
            matched_keyword = keyword
            break
   
    return matched_keyword

async def find_similar_todos_mongodb(query_embedding: List[float], top_k: int = 3):
    """
    Use MongoDB's vector search to find similar todos
    """
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 50,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "priority": 1,
                    "completed": 1,
                    "createdAt": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        similar_todos = await db.todos.aggregate(pipeline).to_list(length=top_k)
        
        # Filter out low similarity results and serialize
        filtered_todos = []
        for todo in similar_todos:
            if todo.get('score', 0) > 0.3:
                # Convert ObjectId to string for each todo
                filtered_todos.append(serialize_todo(todo))
        
        return filtered_todos
    
    except Exception as e:
        print(f"Vector search error: {e}")
