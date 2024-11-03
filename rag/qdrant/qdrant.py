# qdrant/qdrant.py
from qdrant_client import QdrantClient
from rag.config.settings import settings

qdrant_client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    api_key=settings.QDRANT_API_KEY
)