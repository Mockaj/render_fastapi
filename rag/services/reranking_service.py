from typing import List
import voyageai
from rag.config.settings import settings
from rag.models.types import RelevantDocument


def rerank_documents(query: str, documents: List[RelevantDocument], top_k: int = 3) -> List[RelevantDocument]:
    """Rerank documents using Voyage AI reranker-2-lite"""

    # Initialize Voyage client
    vo = voyageai.Client(api_key=settings.VOYAGE_API_KEY)

    # Prepare documents for reranking
    doc_texts = [doc.paragraph_zneni for doc in documents]

    # Call Voyage AI reranker
    reranking = vo.rerank(
        query=query,
        documents=doc_texts,
        model="rerank-2-lite",
        top_k=top_k
    )

    # Reorder documents based on reranking results
    reranked_docs = []
    for result in reranking.results:
        original_index = result.index
        reranked_docs.append(documents[original_index])

    return reranked_docs
