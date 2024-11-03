# app/stt/routers.py

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from stt.services import transcribe_audio_file, run_pipeline
from stt.models import UploadResponse, TranscriptionResponse, ErrorResponse

router = APIRouter()

UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@router.post('/upload', response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    return {"message": "File uploaded successfully", "path": filepath}

@router.get('/transcribe', response_model=TranscriptionResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def transcribe_audio():
    filename = 'recording.webm'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        transcription = transcribe_audio_file(filepath)
        postprocessed_text = run_pipeline(transcription)
        return {"transcription": postprocessed_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))