from pydantic import BaseModel

class UploadResponse(BaseModel):
    message: str
    path: str

class TranscriptionResponse(BaseModel):
    transcription: str

class ErrorResponse(BaseModel):
    error: str
    