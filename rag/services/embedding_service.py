# services/embedding_service.py
from typing import List
from rag.voyage_embed.embed import get_embeddings

def embed_query(query: str) -> List[float]:
    embeddings = get_embeddings([query], input_type="query")
    if not embeddings:
        raise RuntimeError("Failed to obtain embeddings for the query")
    return embeddings[0]