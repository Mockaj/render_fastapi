# voyage_embed/embed.py
from voyageai import Client as VoyageClient
from typing import List
from rag.config import settings

vo_client = VoyageClient(api_key=settings.VOYAGE_API_KEY)

def get_embeddings(
    texts: List[str],
    input_type: str = "document",
    model: str = "voyage-multilingual-2"
) -> List[List[float]]:
    try:
        result = vo_client.embed(texts, model=model, input_type=input_type)
        return result.embeddings
    except Exception as e:
        # Log the error or handle it appropriately
        raise RuntimeError(f"Failed to get embeddings: {e}")