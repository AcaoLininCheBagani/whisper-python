# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel
from datetime import datetime, timezone  # ⚠️ you had 'datatime' typo!
# from piper import text_to_speech
# audio_path = text_to_speech(spoken_response)

import tempfile
import os
import httpx
import json

# >>> ADD MOTOR & DOTENV <<<
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()  # Load .env file

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in environment")

client = AsyncIOMotorClient(MONGO_URI)
db = client.track_my_todo  # your DB name

# FastAPI app
app = FastAPI(title="Voice Todo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- UPGRADED /llm: Reasoning Agent with MongoDB ---
# --- /llm: REASONING AGENT ---
@app.post("/llm")
async def llm_endpoint(request: Request):
    body = await request.json()
    user_input = body.get("text", "").strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Text input is required")

    # Fetch current todos (your schema uses 'title')
    try:
        todos = await db.todos.find(
           
        ).to_list(length=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

    # === 🔍 QUICK RULE: Handle questions WITHOUT LLM ===
    user_lower = user_input.lower()
    question_keywords = ["what", "list", "show", "tell me", "do i have", "anything to do", "my tasks", "todo"]
    is_question = any(kw in user_lower for kw in question_keywords)

    if is_question:
        if not todos:
            response = "Your todo list is empty."
        else:
            total = len(todos)
            pending = [t for t in todos if not t["completed"]]
            response = f"You have {len(pending)} pending tasks."
            # Mention a high-priority one
            high_priority = next((t for t in pending if t.get("priority") == "high"), None)
            if high_priority:
                response += f" Start with: {high_priority['title']}."
        return JSONResponse({"text": response, "raw_llm": "(handled by rule)"})

    # === 🧠 OTHERWISE: Use LLM for command parsing ===
    todo_summary = "\n".join([
        f"- [{'✓' if t['completed'] else ' '}] {t['title']} (priority: {t.get('priority', 'normal')})"
        for t in todos
    ]) or "No tasks."

    prompt = f"""
You are JARVIS, a precise assistant. Today is {datetime.now().strftime('%Y-%m-%d')}.
Current todos:
{todo_summary}

User said: "{user_input}"

Rules:
- If the user gives a clear command to add, complete, or delete a task, respond ONLY with valid JSON.
- The JSON must have two keys: "action" and "item".
- "action" must be exactly one of: "add", "complete", or "delete".
- "item" must be the exact task phrase from the user (no extra words).
- NEVER include explanations, markdown, or placeholders like "add|delete".
- If unsure, respond naturally in under 15 words.

Examples of GOOD output:
User: add buy milk
Response: {{"action": "add", "item": "buy milk"}}

User: complete write report
Response: {{"action": "complete", "item": "write report"}}

User: delete old meeting notes
Response: {{"action": "delete", "item": "old meeting notes"}}

Now respond to this input:
User: "{user_input}"
Response:
""".strip()

    try:
        async with httpx.AsyncClient(timeout=120.0) as ollama:
            resp = await ollama.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_ctx": 1024}
                }
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama error: {resp.text}")
        raw_llm = resp.json().get("response", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # === 🔄 Execute action or return text ===
    spoken_response = raw_llm
    try:
        # Clean potential markdown
        clean_json = raw_llm.replace("```json", "").replace("```", "").strip()
        action_data = json.loads(clean_json)
        action = action_data.get("action")
        item = action_data.get("item")

        if action == "add" and item:
            await db.todos.insert_one({
                "title": item,
                "completed": False,
                "priority": "normal",
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            })
            spoken_response = f"Added: {item}"

        elif action == "complete" and item:
            result = await db.todos.update_one(
                {"title": {"$regex": item, "$options": "i"}},
                {"$set": {"completed": True, "updatedAt": datetime.now(timezone.utc)}}
            )
            spoken_response = "Marked as done." if result.modified_count else "Task not found."

        elif action == "delete" and item:
            result = await db.todos.delete_one(
                {"title": {"$regex": item, "$options": "i"}}
            )
            spoken_response = "Deleted." if result.deleted_count else "Task not found."

    except (json.JSONDecodeError, KeyError, TypeError):
        # Not a valid action → treat as natural response
        pass

    return JSONResponse({
        "text": spoken_response,
        "raw_llm": raw_llm
    })