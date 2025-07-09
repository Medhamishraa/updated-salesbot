# front_end_llm/utils.py
import os
from typing import List, Dict, Optional
from datetime import datetime, UTC # ⭐ IMPORT UTC for consistent timezone handling

from fuzzywuzzy import fuzz

# Import models from the consolidated pydantic_models.py
from front_end_llm.pydantic_models import AskInput, Message, Chat

# Import prompts from the canonical prompts.py
from front_end_llm.prompts import SYSTEM_PROMPT, RETRY_PROMPT_SUFFIX, NEXT_QUESTION_PROMPT


# --------------------
# Filtering Functions
# --------------------

forbidden_phrases = [
    "expected demand", "future demand", "market forecast", "how much future demand",
    "how much demand", "estimate future sales", "foresee any increase in demand",
    "market size", "current market size", "future market size"
]

def is_forbidden(question: str) -> bool:
    return any(phrase in question.lower() for phrase in forbidden_phrases)

def is_duplicate(question: str, qa_items: List[Message], threshold=80) -> bool: # Change type hint
    """
    Checks for duplicate questions based on fuzzy string matching.
    `qa_items` should be a list of Message objects.
    """
    for item in qa_items:
        if item.sender.lower() == "assistant": # Access directly as it's a Message object
            similarity = fuzz.ratio(item.content.lower(), question.lower()) # Access directly
            if similarity >= threshold:
                return True
    return False


# --------------------
# Mongo Functions
# --------------------

def store_message(
    messages_collection,
    chat_id: str,
    user_id: str,
    content: str,
    sender: str,
    message_type: str = "text",
    timestamp: Optional[datetime] = None
) -> Message:
    """Stores a message in the messages collection."""
    # ⭐ Use the provided timestamp or current UTC time
    message_timestamp = timestamp if timestamp is not None else datetime.now(UTC)

    message_data = {
        "chatId": chat_id,
        "userId": user_id,
        "content": content,
        "timestamp": message_timestamp, # ⭐ Use the determined timestamp here
        "sender": sender,
        "message_type": message_type
    }
    result = messages_collection.insert_one(message_data)
    
    # When creating a Message Pydantic model from a MongoDB insertion,
    # you need to correctly map MongoDB's _id to your Pydantic 'id' field
    # and ensure all expected fields are present.
    # The simplest way is to fetch it back, but let's try to construct it if possible.
    
    # We should return a Message instance. Message model expects 'id', not '_id'.
    # Convert ObjectId to string for 'id' field in Pydantic model.
    return Message(
        id=str(result.inserted_id), # ⭐ Map _id to id
        chatId=chat_id,
        userId=user_id,
        content=content,
        timestamp=message_timestamp,
        sender=sender,
        message_type=message_type # Ensure message_type is passed if it's a required field in Message model
    )


def get_chat_messages(
    messages_collection,
    chat_id: str
) -> List[Message]:
    """Retrieves messages for a given chat ID."""
    messages_data = messages_collection.find({"chatId": chat_id}).sort("timestamp", 1)
   
    return [Message(**{**msg, 'id': str(msg['_id'])}) for msg in messages_data]


def get_qa_history_for_llm(messages_collection, chat_id: str) -> List[Dict]:
    """Retrieves QA history specifically formatted for LLM input."""
    messages = get_chat_messages(messages_collection, chat_id)
    history = []
    for msg in messages:
        # Standardize roles for LLM consumption
        if msg.sender == "user":
            history.append({"role": "user", "content": msg.content})
        elif msg.sender == "assistant" or msg.sender == "bot": # Account for "bot" sender
            history.append({"role": "assistant", "content": msg.content})
    return history


# --------------------
# Utility Function
# --------------------

def build_history(qa_items: List[Message]) -> List[Dict[str, str]]:
    """Builds a simplified history from Message objects for LLM context."""
    history = []
    for item in qa_items:
        if item.sender == "user":
            history.append({"role": "user", "content": item.content})
        elif item.sender in {"assistant", "bot"}:
            history.append({"role": "assistant", "content": item.content})
    return history
