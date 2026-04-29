"""OpenAI-compatible STT server using faster-whisper directly."""

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

logger = logging.getLogger("whisper-stt")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("VOXBOX_HF_REPO_ID", "Systran/faster-whisper-large-v3")
DEVICE = os.getenv("VOXBOX_DEVICE", "cpu")
COMPUTE_TYPE = "int8" if DEVICE == "cpu" else "float16"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    from faster_whisper import WhisperModel
    logger.info("Loading model %s on %s (%s)...", MODEL_ID, DEVICE, COMPUTE_TYPE)
    model = WhisperModel(MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("Model loaded.")
    yield


app = FastAPI(title="Whisper STT Server", lifespan=lifespan)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    prompt: Optional[str] = Form(None),
):
    if globals()["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = globals()["model"].transcribe(
            tmp_path,
            language=language or None,
            beam_size=5,
            initial_prompt=prompt,
        )
        text = " ".join(seg.text for seg in segments).strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return JSONResponse({
            "text": text,
            "language": info.language,
            "duration": info.duration,
        })
    return JSONResponse({"text": text})


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "openai"}],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": globals()["model"] is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
