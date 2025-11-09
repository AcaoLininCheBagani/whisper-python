# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
from langchain_ollama import OllamaLLM
from rag_functions import build_context_from_todos, is_date_query, is_yesterday, get_user_intent, get_todos_created_yesterday, get_todos_created_today, find_similar_todos_mongodb
from mongo_connection import db
import re
import numpy as np
import tempfile
import os
import httpx
import json
import asyncio
import uuid
import edge_tts

# MongoDB & Environment
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId  # Add this import

embModel = SentenceTransformer('all-MiniLM-L6-v2')

# Custom JSON encoder for ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Todo item schema
class TodoItem(BaseModel):
    title: str
    completed: bool = False
    priority: str = "medium"

class TodoItemWithEmbedding(TodoItem):
    embedding: Optional[List[float]] = None

# FastAPI app
app = FastAPI(title="Voice Todo API")
llm = OllamaLLM(model="llama3.2:1b")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# API Routes
@app.post("/todos")
async def create_todo(todo: TodoItem):
    try:
        embedding = embModel.encode([todo.title])[0].tolist()
        todo_document = {
            "title": todo.title,
            "completed": todo.completed,
            "priority": todo.priority,
            "embedding": embedding,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        }

        result = await db.todos.insert_one(todo_document)
        if result:
            return {
                "message": "Successfully created todo"
            }
        else:
            raise HTTPException(status_code=400, detail="Error creating todo")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating todo: {str(e)}")

# --- RAG IMPLEMENTATION WITH MONGODB VECTOR SEARCH ---
@app.post("/llm")
async def handle_llm(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    intent = get_user_intent(text)
    try:
        if intent:
            query_embedding = embModel.encode([text])[0].tolist()
            if intent == 'add':
                todos = await get_todos_created_today()
#                 enhanced_prompt = f"""You are a task extractor. Remove unnecessary words like "add". from the following text and return only the core task.

# Text: {text}

# Return only the cleaned text without any additional explanation."""
            enhanced_prompt = f"""You are a task extraction assistant. Analyze the input text and extract only the core actionable task by:

            1. Removing all request phrases like "can you", "please", "add", "I need you to", etc.
            2. Eliminating unnecessary filler words and polite modifiers
            3. Keeping only the essential action + object combination
            4. Ensuring the output is concise and direct
            5. Removing any question formatting or conversational markers

            Input text: {text}

            Return ONLY the cleaned task text with no additional explanations, quotes, or formatting."""
        elif is_date_query(text):
            todos = await get_todos_created_today()
            context = build_context_from_todos(todos)
            enhanced_prompt = f"""
            The user is asking about todos created today. Here are their todos created today:
            
            TODAY'S TODOS:
            {context}
            
            USER QUERY: {text}
            
            Provide a helpful response about what they created today.
            If there are no todos today, suggest creating some.
            
            RESPONSE:
            """
        elif is_yesterday(text):
            todos = await get_todos_created_yesterday()
            context = build_context_from_todos(todos)
            enhanced_prompt = f""" 
            The user is asking about todos created yesterday
            YESTERDAY'S TODOS:
            {context}
            
            can you respond on what todos the user created yesterday based on the context? {text}
            RESPONSE:
            """
        else:
            # Step 1: Generate embedding for the query
            query_embedding = embModel.encode([text])[0].tolist()
            
            # Step 2: Find similar todos using MongoDB vector search
            todos = await find_similar_todos_mongodb(query_embedding, top_k=3)
            
            # Step 3: Build context from similar todos
            context = build_context_from_todos(todos)
            
            # Step 4: Create enhanced prompt with context
            enhanced_prompt = f"""
            Answer the question based only on the following context:
            {context}

            ---
            Answer the question based on the above context: {text}.
            """
        
        # Step 5: Invoke LLM with enhanced prompt
        result = llm.invoke(enhanced_prompt)
        
        return JSONResponse({ 
            "message": result,
            "relevant_todos": todos,
            "query_type": "date_based" if is_date_query(text) | is_yesterday(text) else "semantic_search"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing LLM request: {str(e)}")



# Load Whisper
print("🚀 Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Whisper model ready!")

# --- /transcribe unchanged ---
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload audio.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        segments, info = model.transcribe(tmp_path, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()

        return JSONResponse({
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
@app.post("/tts")
async def text_to_speech(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")

    voice = body.get("voice", "en-AU-WilliamNeural")  # Default to your preferred voice
    valid_voices = ['en-AU-NatashaNeural', 'en-AU-WilliamNeural']
    if voice not in valid_voices:
        voice = "en-AU-WilliamNeural"

    # Generate unique filename
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(tempfile.gettempdir(), filename)

    try:
        # Run edge_tts asynchronously
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

        # Return the file path (client can fetch it via /audio/{filename} if needed)
        return JSONResponse({
            "message": "Audio generated successfully",
            "audio_file": filename,  # Just the filename for reference
            "path": output_path       # Full path if server needs to serve it
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
    
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/mpeg")