# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from db_utils.chathistory_db import store_chat_message, get_chat_history

app = FastAPI()

class ChatMessage(BaseModel):
    session_uuid: str
    question: str
    answer: str
    role: str = "user"

@app.post("/chat")
async def save_chat(msg: ChatMessage):
    try:
        result = store_chat_message(
            session_uuid=msg.session_uuid,
            question=msg.question,
            answer=msg.answer,
            role=msg.role
        )
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/{session_uuid}")
async def fetch_chat(session_uuid: str):
    try:
        history = get_chat_history(session_uuid)
        if history:
            return history
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
