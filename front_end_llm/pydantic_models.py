# front_end_llm/pydantic_models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any,Union
from datetime import datetime
from bson import ObjectId

# ⭐ UPDATED PyObjectId class for Pydantic v2
class PyObjectId(ObjectId):
    """
    Custom type for MongoDB's ObjectId to ensure Pydantic can handle it.
    It provides validators for both string and ObjectId input and allows
    Pydantic to serialize it to a string for JSON output.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any, info) -> ObjectId:
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    # ⭐ THIS IS THE CHANGE: Replaced __modify_schema__ with __get_pydantic_json_schema__
    # This method is used by Pydantic v2 to define how the type appears in JSON schemas (e.g., OpenAPI docs)
    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: dict, handler
    ) -> dict: # Note: handler is the SchemaOrFieldHandler, core_schema is the current schema
        # Modify the core_schema to represent ObjectId as a string
        # The exact structure of core_schema might vary, but we want to ensure
        # that the resulting schema for this field is of type "string".
        # A simple way for external types is to just return a basic string schema.
        return handler.generate_schema(str) # Tell Pydantic to treat this as a string in the schema

# ⭐ ENSURE THESE ARE PRESENT:
class GoogleAuthRequest(BaseModel):
    token: str

class FacebookAuthRequest(BaseModel):
    accessToken: str # This is what the frontend will send after successful FB login

# --- LLM Specific Model ---
class AskInput(BaseModel):
    prompt: str
    history: List[Dict[str, str]]
    qa_items: List[Dict[str, str]]

# --- User Schemas (Moved from schemas.py) ---
class UserLoginRequest(BaseModel):
    email: str
    password: str
    captchaToken: str

class UserSignupRequest(BaseModel):
    firstName: str
    lastName: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: PyObjectId = Field(alias="_id")
    firstName: str
    lastName: str
    email: str
    profile: Optional[Dict[str, Any]] = None # <--- This field

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

# --- Message Schemas (Moved from schemas.py) ---
class Message(BaseModel):
    id: PyObjectId = Field(alias="_id", default_factory=PyObjectId)
    chatId: PyObjectId # Assuming chatId is also an ObjectId
    userId: str
    content: str
    timestamp: datetime
    sender: str

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class SendMessageRequest(BaseModel):
    content: str

# --- Chat Schemas (Moved from schemas.py) ---
class Chat(BaseModel):
    id: PyObjectId = Field(alias="_id", default_factory=PyObjectId)
    participants: List[str]
    lastMessage: Optional[str] = None
    lastActivity: datetime
    title: str

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class CreateChatRequest(BaseModel):
    type: str = "direct"
    title: Optional[str] = None

# --- API Responses (Moved from schemas.py) ---
class SuccessMessageResponse(BaseModel):
    success: bool = True
    message: str

class AuthSuccessResponse(BaseModel):
    success: bool = True
    token: str
    user: UserResponse
    message: str

class ChatSuccessResponse(BaseModel):
    success: bool = True
    chat: Chat
    message: str

class ChatsListResponse(BaseModel):
    success: bool = True
    chats: List[Chat]


class ConversationTurn(BaseModel):
    # Represents a single question-answer turn in the chat
    question: str
    answer: str
    timestamp: datetime
    role: str # "user" or "assistant" - indicates whose message is primary for the turn

class MessageSuccessResponse(BaseModel):
    success: bool = True
    message: Union[Message, ConversationTurn]

class MessagesListResponse(BaseModel):
    success: bool = True
    messages: List[Message]

class ErrorResponse(BaseModel):
    success: bool = False
    message: str

# --- NEW MODELS FOR TRANSFORMED MESSAGE OUTPUT ---


class TransformedChatMessagesResponse(BaseModel):
    # Change the internal field name from _id to id, and set the alias to "_id"
    id: str = Field(alias="_id") # ⭐ FIXED LINE HERE
    session_uuid: str
    messages: List[ConversationTurn]

    # You also need to ensure that the ConfigDict is set for this model
    # to allow population by name and to use the alias correctly.
    model_config = ConfigDict(
        populate_by_name=True, # Allow field assignment by both the actual name ('id') and alias ('_id')
        arbitrary_types_allowed=True, # Needed if you have custom types that Pydantic doesn't recognize
        json_encoders={ObjectId: str} # Ensure ObjectId is serialized to string
    )