from fastapi import FastAPI, Request, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

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

# Add this middleware for longer timeouts
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=60.0)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "Request timeout"}
        )

routers_prefix = "/api/v1"

# Include routers with prefixes
app.include_router(stt_router, prefix=f"{routers_prefix}/stt")
app.include_router(rag_router, prefix=f"{routers_prefix}/rag")