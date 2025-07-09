# db_utils/dpt_db.py

from datetime import datetime
from db_utils.db_config import get_collection

DB_NAME = "company"
COLLECTION_NAME = "company_data"

collection = get_collection(DB_NAME, COLLECTION_NAME)

def insert_agent_result(query: str, result: str):
    """
    Inserts a query and result into MongoDB.
    """
    document = {
        "query": query,
        "result": result,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(document)
    print("[✅] Agent result stored in MongoDB.")

def get_all_results():
    """
    Retrieves all stored agent results.
    """
    return list(collection.find({}, {"_id": 0}))
