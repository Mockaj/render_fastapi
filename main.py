from fastapi import FastAPI, Request, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
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

# Custom middleware for handling long-running requests


class LongRunningMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=120.0)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timeout"}
            )


app.add_middleware(LongRunningMiddleware)

routers_prefix = "/api/v1"

# Include routers with prefixes
app.include_router(stt_router, prefix=f"{routers_prefix}/stt")
app.include_router(rag_router, prefix=f"{routers_prefix}/rag")
