"""OpenAI-compatible TTS server backed by edge-tts (supports Thai and 400+ voices)."""

import time
import uvicorn
import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Edge TTS Server")

DEFAULT_VOICE = "th-TH-PremwadeeNeural"


class SpeechRequest(BaseModel):
    model: str = "edge-tts"
    input: str
    voice: str = DEFAULT_VOICE
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if not req.input:
        raise HTTPException(status_code=400, detail="input is required")

    rate = f"{int((req.speed - 1.0) * 100):+d}%"
    communicate = edge_tts.Communicate(req.input, req.voice, rate=rate)

    async def generate():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(generate(), media_type="audio/mpeg")


@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [{"id": "edge-tts", "object": "model", "created": int(time.time()), "owned_by": "microsoft"}],
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8881)
