from datetime import datetime
from db_utils.db_config import get_collection
import os

DB_NAME = os.getenv("CHAT_DB_NAME")
COLLECTION_NAME = os.getenv("CHAT_COLLECTION_NAME")

collection = get_collection(DB_NAME, COLLECTION_NAME)

def store_chat_message(session_uuid: str, question: str, answer: str, role: str = "user"):
    message = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow(),
        "role": role
    }

    existing = collection.find_one({"session_uuid": session_uuid})
    if existing:
        collection.update_one(
            {"session_uuid": session_uuid},
            {"$push": {"messages": message}}
        )
        return "Message added to existing session"
    else:
        collection.insert_one({
            "session_uuid": session_uuid,
            "messages": [message],
            "created_at": datetime.utcnow()
        })
        return "New session created"

def get_chat_history(session_uuid: str):
    return collection.find_one({"session_uuid": session_uuid}, {"_id": 0})

