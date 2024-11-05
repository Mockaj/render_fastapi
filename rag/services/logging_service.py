from typing import List
from rag.db.models import SemanticSearchLog
from rag.db.config import supabase_client
from rag.logger import logger
from rag.models.types import RelevantDocument

async def log_semantic_search(query: str, results: List[RelevantDocument]) -> None:
    """
    Logs semantic search results to Supabase.
    
    Args:
        query: The search query
        results: List of relevant documents returned from the search
    """
    try:
        logs = []
        for result in results:
            log = SemanticSearchLog(
                query=query,
                law_name=result.law_nazev,
                paragraph=int(result.paragraph_cislo),
                law_year=int(result.law_year),
                law_id=int(result.law_id),
                score=result.score if hasattr(result, 'score') else 0.0
            )
            logs.append(log.model_dump(exclude={'id', 'created_at'}))

        if logs:
            response = supabase_client.table('semantic_search_logs').insert(logs).execute()
            
            if hasattr(response, 'error') and response.error is not None:
                logger.error(f"Failed to log semantic search results: {response.error}")
            else:
                logger.info(f"Successfully logged {len(logs)} semantic search results")
                
    except Exception as e:
        logger.error(f"Error logging semantic search results: {e}")
        # Don't raise the exception - we don't want to break the main flow if logging fails 