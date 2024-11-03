from pymongo import MongoClient
from typing import List
from rag.logger.logger import logger
from rag.models import Law


def get_mongo_client(uri: str = "mongodb://localhost:27017/") -> MongoClient:
    return MongoClient(uri)

def fetch_laws_from_mongodb(db_name: str = "law_database_mvp", collection_name: str = "MVP") -> List[Law]:
    client = get_mongo_client()
    db = client[db_name]
    collection = db[collection_name]
    laws_data = list(collection.find())
    laws = []
    for law_data in laws_data:
        try:
            # Remove MongoDB '_id' field
            law_data.pop('_id', None)
            # Build Law object
            law = Law(**law_data)
            laws.append(law)
        except Exception as e:
            logger.error(f"Failed to parse law data: {e}")
    return laws