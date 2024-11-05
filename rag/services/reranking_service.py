from typing import List
import voyageai
from rag.config.settings import settings
from rag.models.types import RelevantDocument


def rerank_documents(query: str, documents: List[RelevantDocument], top_k: int = 3) -> List[RelevantDocument]:
    """Rerank documents using Voyage AI reranker-2"""

    # Initialize Voyage client
    vo = voyageai.Client(api_key=settings.VOYAGE_API_KEY)

    # Prepare documents for reranking
    doc_texts = [doc.paragraph_zneni for doc in documents]

    # Call Voyage AI reranker
    reranking = vo.rerank(
        query=query,
        documents=doc_texts,
        model="rerank-2",
        top_k=top_k
    )

    # Reorder documents based on reranking results and add scores
    reranked_docs = []
    for result in reranking.results:
        original_index = result.index
        doc = documents[original_index]
        # Create a new document with the score
        doc_with_score = RelevantDocument(
            law_nazev=doc.law_nazev,
            law_id=doc.law_id,
            law_year=doc.law_year,
            law_category=doc.law_category,
            law_date=doc.law_date,
            law_staleURL=doc.law_staleURL,
            paragraph_cislo=doc.paragraph_cislo,
            paragraph_zneni=doc.paragraph_zneni,
            score=result.relevance_score
        )
        reranked_docs.append(doc_with_score)

    return reranked_docs
