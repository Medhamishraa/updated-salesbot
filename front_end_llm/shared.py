from typing import List, Dict
from fuzzywuzzy import fuzz

# Forbidden phrase list
forbidden_phrases = [
    "expected demand", "future demand", "market forecast", "how much future demand",
    "how much demand", "estimate future sales", "foresee any increase in demand",
    "market size", "current market size", "future market size"
]

def is_forbidden(question: str) -> bool:
    return any(phrase in question.lower() for phrase in forbidden_phrases)

def is_duplicate(question: str, qa_items: List[Dict], threshold=80) -> bool:
    for item in qa_items:
        if item["role"] == "assistant":
            similarity = fuzz.ratio(item["question"].lower(), question.lower())
            if similarity >= threshold:
                return True
    return False

def get_chat_messages(qa_items: List[Dict]) -> List[Dict[str, str]]:
    history = []
    for item in qa_items:
        if item["role"] == "assistant":
            history.append({"role": "assistant", "content": item["question"]})
        elif item["role"] == "user":
            history.append({"role": "user", "content": item["answer"]})
    return history

def get_qa_history_for_llm(chat: List[Dict]) -> List[Dict[str, str]]:
    # Can be adjusted if needed later
    return get_chat_messages(chat)
