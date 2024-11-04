from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stt.routers import router as stt_router
from rag.routers.context import router as rag_router

app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this according to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

routers_prefix = "/api/v1"

# Include routers with prefixes
app.include_router(stt_router, prefix=f"{routers_prefix}/stt")
app.include_router(rag_router, prefix=f"{routers_prefix}/rag")
