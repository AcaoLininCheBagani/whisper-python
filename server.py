from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel
import tempfile
import os
import httpx

app = FastAPI(title="Voice Todo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
print("🚀 Loading Whisper model (this may take a few seconds)...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Whisper model ready!")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Validate file
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload audio.")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe
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
        # Clean up temp file
        if 'tmp_path' in locals():
            os.unlink(tmp_path)


@app.post("/llm")
async def llm_endpoint(request: Request):
    body = await request.json()
    user_input = body.get("text", "").strip()
        
    prompt = f"""
    You are a precise command parser. Convert the user's input into a JSON object with exactly two keys: "action" and "item".

    Rules:
    - "action" must be one of: "add", "delete", "complete", "edit"
    - "item" must be a short, clean phrase (no extra words)
    - Output ONLY valid JSON. No explanations. No markdown. No code blocks.

    Examples:
    Input: "add buy milk"
    Output: {{"action": "add", "item": "buy milk"}}

    Input: "delete jogging todo"
    Output: {{"action": "delete", "item": "jogging"}}

    Input: "complete the report"
    Output: {{"action": "complete", "item": "the report"}}

    Now parse this:
    Input: "{user_input}"
    Output:
    """.strip()
    try:
        # Parse JSON body
        # body = await request.json()
        # prompt = body.get("prompt", "").strip()
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required and cannot be empty")

        # Call Ollama (running on your Mac at localhost:11434)
        async with httpx.AsyncClient(timeout=120.0) as client:
            ollama_response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:1b",      # ← You can change this!
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,      # Lower = more deterministic
                        "num_ctx": 512           # Context window
                    }
                }
            )

        # Handle Ollama errors
        if ollama_response.status_code != 200:
            error_detail = ollama_response.text or "Unknown Ollama error"
            raise HTTPException(status_code=500, detail=f"Ollama error: {error_detail}")

        # Parse and return clean response
        ollama_data = ollama_response.json()
        clean_response = ollama_data.get("response", "").strip()

        return JSONResponse({
            "response": clean_response
        })

    except Exception as e:
        print(f"LLM endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")