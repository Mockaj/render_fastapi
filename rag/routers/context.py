from fastapi import APIRouter, Depends, HTTPException, Body, Query, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from rag.models.types import QueryRequest, QueryResponse, RelevantDocument, SeedLawRequest
from rag.services.langtail_service import enhance_query_with_langtail
from rag.services.embedding_service import embed_query
from rag.services.qdrant_service import search_qdrant
from rag.config.settings import settings
from rag.services.auth_service import get_current_username
import secrets
from rag.models import Law
from rag.services.seed_service import seed_law_from_url
from rag.services.reranking_service import rerank_documents

router = APIRouter()

security = HTTPBasic()

@router.post("/context", response_model=QueryResponse)
async def get_context(
    request: QueryRequest = Body(...),
    n: int = Query(default=settings.DEFAULT_N, ge=1),
    username: str = Depends(get_current_username)
):
    try:
        # Enhance the query using Langtail
        enhanced_query = enhance_query_with_langtail(request.query)

        # Embed the enhanced query
        embedding = embed_query(enhanced_query)

        # First get 5 documents from Qdrant
        initial_documents = search_qdrant(embedding=embedding, top_n=5)

        # Rerank the documents
        reranked_documents = rerank_documents(
            query=request.query,
            documents=initial_documents,
            top_k=n
        )

        return QueryResponse(relevant_docs=reranked_documents)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))

@router.post("/seed", response_model=Law)
async def seed_law(
    request: SeedLawRequest = Body(...),
    username: str = Depends(get_current_username)
):
    try:
        law = await seed_law_from_url(request.url)
        return law
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
