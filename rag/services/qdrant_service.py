# services/qdrant_service.py
from qdrant_client.http.models import Filter, SearchRequest, PointStruct
from typing import List
from rag.qdrant.qdrant import qdrant_client
from rag.config.settings import settings
from rag.models.types import RelevantDocument

def search_qdrant(embedding: List[float], top_n: int) -> List[RelevantDocument]:
    try:
        results = qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=embedding,
            limit=top_n
        )
        documents = []
        for result in results:
            payload = result.payload
            document = RelevantDocument(
                law_nazev=payload.get('law_nazev'),
                law_id=payload.get('law_id'),
                law_year=payload.get('law_year'),
                law_category=payload.get('law_category'),
                law_date=payload.get('law_date'),
                law_staleURL=payload.get('law_staleURL'),
                paragraph_cislo=payload.get('paragraph_cislo'),
                paragraph_zneni=payload.get('paragraph_zneni')
            )
            documents.append(document)
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to search Qdrant: {e}")