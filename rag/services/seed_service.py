import re
from typing import Tuple, List
from rag.models import Law
from rag.utils.open_data_to_mongo import get_law_details
from rag.config.settings import settings
from rag.voyage_embed import get_embeddings
from rag.qdrant import qdrant_client
from rag.logger import logger
from qdrant_client.models import PointStruct

BATCH_SIZE = 100


def parse_law_url(url: str) -> Tuple[str, str]:
    match = re.search(r'/cs/(\d{4})-(\d+)$', url)
    if not match:
        raise ValueError("Invalid URL format")
    year, law_id = match.groups()
    return year, law_id


def process_batch(texts: List[str], payloads: List[dict], start_id: int) -> None:
    try:
        embeddings = get_embeddings(texts)

        # Create points for this batch
        points = [
            PointStruct(
                id=start_id + i,
                vector=embedding,
                payload=payload
            )
            for i, (embedding, payload) in enumerate(zip(embeddings, payloads))
        ]

        # Upload batch to Qdrant
        qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            wait=True,
            points=points
        )
    except Exception as e:
        logger.error(f"Failed to process batch: {e}")
        raise


async def seed_law_from_url(url: str) -> Law:
    # Parse the URL
    year, law_id = parse_law_url(url)

    # Create initial Law object
    law = Law(
        nazev="",  # Will be filled by get_law_details
        id=law_id,
        year=year
    )

    # Get full law details from API
    law = get_law_details(law, settings.OPEN_DATA_API_KEY)

    # Get current max ID from Qdrant to avoid conflicts
    points = qdrant_client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        limit=1,
        with_payload=False,
        with_vectors=False
    )[0]

    next_id = max([p.id for p in points], default=0) + 1

    # Process paragraphs in batches
    current_batch_texts = []
    current_batch_payloads = []

    for para in law.paragrafy:
        if not para.zneni.strip():
            continue

        current_batch_texts.append(para.zneni)
        current_batch_payloads.append({
            "law_nazev": law.nazev,
            "law_id": law.id,
            "law_year": law.year,
            "law_category": law.category,
            "law_date": law.date,
            "law_staleURL": law.staleURL,
            "paragraph_cislo": para.cislo,
            "paragraph_zneni": para.zneni,
        })

        # Process batch if it reaches the size limit
        if len(current_batch_texts) >= BATCH_SIZE:
            process_batch(
                current_batch_texts,
                current_batch_payloads,
                next_id
            )
            next_id += len(current_batch_texts)
            current_batch_texts = []
            current_batch_payloads = []

    # Process any remaining items
    if current_batch_texts:
        process_batch(
            current_batch_texts,
            current_batch_payloads,
            next_id
        )

    return law
