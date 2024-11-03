from rag.logger import logger
from rag.mongo import fetch_laws_from_mongodb
from rag.qdrant import qdrant_client
from qdrant_client.models import PointStruct, Distance, VectorParams
from tqdm import tqdm
from dotenv import load_dotenv
from rag.voyage_embed import get_embeddings
from rag.models import Law, Paragraf
load_dotenv()


def main() -> None:
    # Fetch laws from MongoDB
    logger.info("Fetching laws from MongoDB")
    laws: list[Law] = fetch_laws_from_mongodb()

    if not laws:
        logger.error("No laws found in MongoDB")
        return

    # Name of the collection
    collection_name = "LAWS_MVP"

    # Determine vector size
    logger.info("Determining vector size")
    sample_text: str = "Sample text to determine vector size"
    sample_embedding: list[list[float]] = get_embeddings([sample_text])
    if not sample_embedding:
        logger.error("Failed to get sample embedding to determine vector size")
        return
    vector_size: int = len(sample_embedding[0])
    logger.info(f"Vector size determined: {vector_size}")

    if qdrant_client.collection_exists(collection_name=collection_name):
        logger.info(f"Collection '{collection_name}' exists in Qdrant proceeding")
    else:
        logger.info(f"Creating collection '{collection_name}' in Qdrant")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    point_id: int= 0  # Unique ID for each point
    batch_size: int= 100  # Adjust as needed
    texts_to_embed: list[str] = []
    point_ids: list[int] = []
    payloads: ist[dict[str, str | None]] = []

    for law in tqdm(laws, desc="Processing laws"):
        for para in law.paragrafy:
            paragraph_text = para.zneni
            if not paragraph_text.strip():
                continue  # Skip empty paragraphs

            # Prepare the text for embedding
            texts_to_embed.append(paragraph_text)
            point_ids.append(point_id)

            # Prepare payload with all the info
            payload = {
                "law_nazev": law.nazev,
                "law_id": law.id,
                "law_year": law.year,
                "law_category": law.category,
                "law_date": law.date,
                "law_staleURL": law.staleURL,
                "paragraph_cislo": para.cislo,
                "paragraph_zneni": para.zneni,
            }
            payloads.append(payload)
            point_id += 1

            # If batch size is reached, process embeddings and upload
            if len(texts_to_embed) >= batch_size:
                embeddings = get_embeddings(texts_to_embed)
                if len(embeddings) != len(texts_to_embed):
                    logger.error("Number of embeddings does not match number of texts")
                else:
                    # Build point structs
                    batch_points = [
                        PointStruct(
                            id=pid,
                            vector=embedding,
                            payload=pl
                        ) for pid, embedding, pl in zip(point_ids, embeddings, payloads)
                    ]
                    # Upload to Qdrant
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=batch_points
                    )
                # Reset lists
                texts_to_embed = []
                point_ids = []
                payloads = []

    # Process any remaining texts
    if texts_to_embed:
        embeddings = get_embeddings(texts_to_embed)
        if len(embeddings) != len(texts_to_embed):
            logger.error("Number of embeddings does not match number of texts")
        else:
            batch_points = [
                PointStruct(
                    id=pid,
                    vector=embedding,
                    payload=pl
                ) for pid, embedding, pl in zip(point_ids, embeddings, payloads)
            ]
            qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=batch_points
            )

if __name__ == "__main__":
    main()