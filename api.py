import os
import certifi
import httpx
from openai import OpenAI
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime, timedelta, UTC

from typing import List, Optional, Dict, Any
import asyncio

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient

import jwt
from fastapi import Header, Depends, HTTPException, status, Request, FastAPI, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-jwt-key-please-change-me")
BOT_USER_ID = "ungli_bot_system_id_12345"
BOT_SENDER_NAME = "bot"

# JWT Helpers
def generate_token(user_id: str, email: str):
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.now(UTC) + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        print("Token has expired.")
        return None
    except jwt.InvalidTokenError:
        print("Invalid token.")
        return None

async def get_current_user_id(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization token is missing!")
    token = authorization.split(" ")[1] if "Bearer" in authorization else authorization
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token is invalid or expired!")
    return payload['user_id']

MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found. Please set it in .env file.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env file.")

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8081",
]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SessionMiddleware, secret_key="supersecretkey")

@app.on_event("startup")
async def startup_db_client():
    app.state.mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    app.state.db = app.state.mongo_client["chatSaaS"]
    app.state.users_collection = app.state.db["users_creds"]
    app.state.chats_collection = app.state.db["chats"]
    app.state.messages_collection = app.state.db["messages"]
    app.state.openai_client = OpenAI(api_key=OPENAI_API_KEY)

@app.on_event("shutdown")
async def shutdown_db_client():
    if hasattr(app.state, 'mongo_client') and app.state.mongo_client:
        app.state.mongo_client.close()

from front_end_llm.pydantic_models import (
    UserLoginRequest, UserSignupRequest, UserResponse,
    SendMessageRequest, CreateChatRequest,
    SuccessMessageResponse, AuthSuccessResponse, ChatSuccessResponse,
    ChatsListResponse, MessageSuccessResponse, MessagesListResponse, ErrorResponse,
    Chat as ChatModel,
    Message as MessageModel,
    GoogleAuthRequest, FacebookAuthRequest, ConversationTurn,
    TransformedChatMessagesResponse
)

from front_end_llm.utils import (
    store_message as llm_store_message,
    get_qa_history_for_llm,
    build_history,
)
from front_end_llm.front_end_llm import ask_openai
from front_end_llm.prompts import SYSTEM_PROMPT, RETRY_PROMPT_SUFFIX, NEXT_QUESTION_PROMPT

# === NEXT CHUNK: Routes start from here ===

@app.get("/api/health", response_model=SuccessMessageResponse)
async def health_check():
    return SuccessMessageResponse(message="healthy")
# --- Auth Endpoints ---

@app.post("/api/auth/signup", response_model=AuthSuccessResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def signup(signup_data: UserSignupRequest, request: Request):
    users_collection_dep = request.app.state.users_collection
    
    if not all([signup_data.firstName, signup_data.lastName, signup_data.email, signup_data.password]):
        raise HTTPException(status_code=400, detail="All fields are required")

    if users_collection_dep.find_one({"email": signup_data.email}):
        raise HTTPException(status_code=400, detail="User already exists with this email")

    hashed_password = generate_password_hash(signup_data.password)
    user_data = {
        "firstName": signup_data.firstName,
        "lastName": signup_data.lastName,
        "email": signup_data.email,
        "password": hashed_password,
        "createdAt": datetime.now(UTC),
        "social_login_provider": None,
        "profile": {}
    }
    result = users_collection_dep.insert_one(user_data)
    user_id = str(result.inserted_id)
    token = generate_token(user_id, signup_data.email)

    user_response_data = UserResponse(
        _id=user_id,
        firstName=signup_data.firstName,
        lastName=signup_data.lastName,
        email=signup_data.email,
        profile={}
    )

    return AuthSuccessResponse(token=token, user=user_response_data, message="User registered successfully")


@app.post("/api/auth/login", response_model=AuthSuccessResponse, responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def login(login_data: UserLoginRequest, request: Request):
    users_collection_dep = request.app.state.users_collection

    if not login_data.captchaToken:
        raise HTTPException(status_code=400, detail="CAPTCHA token is required")

    user = users_collection_dep.find_one({"email": login_data.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if user.get("social_login_provider"):
        raise HTTPException(status_code=401, detail=f"Please log in with {user['social_login_provider']} for this email.")

    if not check_password_hash(user["password"], login_data.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = generate_token(str(user["_id"]), user["email"])

    user_response_data = UserResponse(
        _id=str(user["_id"]),
        firstName=user["firstName"],
        lastName=user["lastName"],
        email=user["email"],
        profile=user.get("profile", {})
    )

    return AuthSuccessResponse(token=token, user=user_response_data, message=f"Welcome back, {user['firstName']}!")
@app.post("/api/auth/google", response_model=AuthSuccessResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def google_auth(google_auth_data: GoogleAuthRequest, request: Request):
    users_collection_dep = request.app.state.users_collection

    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google client ID not configured on backend.")

    try:
        id_info = id_token.verify_oauth2_token(
            google_auth_data.token, google_requests.Request(), GOOGLE_CLIENT_ID
        )

        email = id_info['email']
        first_name = id_info.get('given_name', '')
        last_name = id_info.get('family_name', '')
        full_name = id_info.get('name', f"{first_name} {last_name}".strip())

        user = users_collection_dep.find_one({"email": email})

        if user:
            user_id = str(user["_id"])
            token = generate_token(user_id, email)
            user_response_data = UserResponse(
                _id=user_id,
                firstName=user.get("firstName", first_name),
                lastName=user.get("lastName", last_name),
                email=email,
                profile={"name": full_name}
            )
            return AuthSuccessResponse(token=token, user=user_response_data, message=f"Welcome back, {user.get('firstName', email)}!")
        else:
            dummy_password = generate_password_hash(f"google_social_{email}_{datetime.now(UTC).timestamp()}")
            user_data = {
                "firstName": first_name,
                "lastName": last_name,
                "email": email,
                "password": dummy_password,
                "createdAt": datetime.now(UTC),
                "social_login_provider": "google",
                "profile": {"name": full_name}
            }
            result = users_collection_dep.insert_one(user_data)
            user_id = str(result.inserted_id)
            token = generate_token(user_id, email)

            user_response_data = UserResponse(
                _id=user_id,
                firstName=first_name,
                lastName=last_name,
                email=email,
                profile={"name": full_name}
            )
            return AuthSuccessResponse(token=token, user=user_response_data, message="Google signup successful!")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Google ID token: {e}")
    except Exception as e:
        print(f"Google authentication error: {e}")
        raise HTTPException(status_code=500, detail="Google authentication failed due to an internal error.")
@app.post("/api/auth/facebook", response_model=AuthSuccessResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def facebook_auth(facebook_auth_data: FacebookAuthRequest, request: Request):
    users_collection_dep = request.app.state.users_collection

    if not FACEBOOK_APP_ID or not FACEBOOK_APP_SECRET:
        raise HTTPException(status_code=500, detail="Facebook app credentials not configured on backend.")

    async with httpx.AsyncClient() as client:
        try:
            graph_api_url = f"https://graph.facebook.com/v19.0/me?fields=id,name,email&access_token={facebook_auth_data.accessToken}"
            
            user_profile_response = await client.get(graph_api_url)
            user_profile_response.raise_for_status()
            user_profile_data = user_profile_response.json()

            email = user_profile_data.get('email')
            if not email:
                raise HTTPException(status_code=400, detail="Could not retrieve a verified email from Facebook. Please ensure your Facebook account has a public email address and you granted email permission.")

            full_name = user_profile_data.get('name', '')
            name_parts = full_name.split(' ', 1)
            first_name = name_parts[0] if name_parts else ''
            last_name = name_parts[1] if len(name_parts) > 1 else ''

            user = users_collection_dep.find_one({"email": email})

            if user:
                user_id = str(user["_id"])
                token = generate_token(user_id, email)
                user_response_data = UserResponse(
                    _id=user_id,
                    firstName=user.get("firstName", first_name),
                    lastName=user.get("lastName", last_name),
                    email=email,
                    profile={"name": full_name}
                )
                return AuthSuccessResponse(token=token, user=user_response_data, message=f"Welcome back, {user.get('firstName', email)}!")
            else:
                dummy_password = generate_password_hash(f"facebook_social_{email}_{datetime.now(UTC).timestamp()}")
                user_data = {
                    "firstName": first_name,
                    "lastName": last_name,
                    "email": email,
                    "password": dummy_password,
                    "createdAt": datetime.now(UTC),
                    "social_login_provider": "facebook",
                    "profile": {"name": full_name}
                }
                result = users_collection_dep.insert_one(user_data)
                user_id = str(result.inserted_id)
                token = generate_token(user_id, email)

                user_response_data = UserResponse(
                    _id=user_id,
                    firstName=first_name,
                    lastName=last_name,
                    email=email,
                    profile={"name": full_name}
                )
                return AuthSuccessResponse(token=token, user=user_response_data, message="Facebook signup successful!")

        except httpx.HTTPStatusError as e:
            print(f"HTTP error during Facebook auth: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Facebook API error: {e.response.text}")
        except Exception as e:
            print(f"Facebook authentication error: {e}")
            raise HTTPException(status_code=500, detail="Facebook authentication failed due to an internal error.")
@app.get("/api/chats", response_model=ChatsListResponse, responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def get_user_chats(
    request: Request,
    # current_user_id: str = Depends(get_current_user_id),  # AUTH BYPASS
):
    current_user_id = "test_user_for_bypass"

    chats_collection_dep = request.app.state.chats_collection

    user_chats_raw = list(chats_collection_dep.find(
        {"participants": current_user_id},
        {"_id": 1, "participants": 1, "lastMessage": 1, "lastActivity": 1, "title": 1}
    ))

    user_chats = []
    for chat_data in user_chats_raw:
        chat_data['_id'] = str(chat_data['_id'])  # Convert ObjectId to str
        user_chats.append(ChatModel(**chat_data))

    return ChatsListResponse(chats=user_chats)
@app.post("/api/chats", response_model=ChatsListResponse, responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def create_chat(
    chat_data: CreateChatRequest,
    request: Request,
    # current_user_id: str = Depends(get_current_user_id),  # AUTH BYPASS
):
    current_user_id = "test_user_for_bypass"

    chats_collection_dep = request.app.state.chats_collection
    messages_collection_dep = request.app.state.messages_collection
    openai_client_dep = request.app.state.openai_client

    chat_title = chat_data.title if chat_data.title is not None else "ungli-untitled"
    chat_type = chat_data.type
    participants = [current_user_id]

    initial_ai_question = "What is the name or model of the product?"
    welcome_message_content = f"Hello! {initial_ai_question}"

    new_chat_doc = {
        "title": chat_title,
        "type": chat_type,
        "participants": participants,
        "createdAt": datetime.now(UTC),
        "lastActivity": datetime.now(UTC),
        "lastMessage": welcome_message_content
    }

    result = chats_collection_dep.insert_one(new_chat_doc)
    new_chat_id = str(result.inserted_id)

    llm_store_message(
        messages_collection=messages_collection_dep,
        chat_id=new_chat_id,
        user_id=BOT_USER_ID,
        content=welcome_message_content,
        sender=BOT_SENDER_NAME,
        timestamp=datetime.now(UTC)
    )

    created_chat_doc = chats_collection_dep.find_one({"_id": ObjectId(new_chat_id)})
    if not created_chat_doc:
        raise HTTPException(status_code=500, detail="Failed to retrieve created chat.")

    created_chat_doc['_id'] = str(created_chat_doc['_id'])

    return ChatsListResponse(
        success=True,
        chats=[ChatModel(**created_chat_doc)]
    )
@app.post("/api/chats/{chat_id}/messages", response_model=MessageSuccessResponse, responses={401: {"model": ErrorResponse}, 400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def post_message(
    chat_id: str,
    message_data: SendMessageRequest,
    request: Request,
    # current_user_id: str = Depends(get_current_user_id),  # AUTH BYPASS
):
    current_user_id = "test_user_for_bypass"

    users_collection_dep = request.app.state.users_collection
    chats_collection_dep = request.app.state.chats_collection
    messages_collection_dep = request.app.state.messages_collection
    openai_client_dep = request.app.state.openai_client

    chat = chats_collection_dep.find_one({"_id": ObjectId(chat_id)})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if not message_data.content:
        raise HTTPException(status_code=400, detail="Message content is required")

    user_message_instance = llm_store_message(
        messages_collection=messages_collection_dep,
        chat_id=chat_id,
        user_id=current_user_id,
        content=message_data.content,
        sender="user"
    )

    chats_collection_dep.update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": {
            "lastMessage": message_data.content,
            "lastActivity": datetime.now(UTC)
        }}
    )

    try:
        bot_response_content = await ask_openai(
            user_message_content=message_data.content,
            chat_id=chat_id,
            user_id=current_user_id,
            openai_client=openai_client_dep,
            messages_collection=messages_collection_dep,
        )
    except Exception as e:
        print(f"Error getting AI response: {e}")
        import traceback
        traceback.print_exc()
        bot_response_content = "I'm having trouble responding right now. Please try again later."

    bot_message_instance = llm_store_message(
        messages_collection=messages_collection_dep,
        chat_id=chat_id,
        user_id=BOT_USER_ID,
        content=bot_response_content,
        sender=BOT_SENDER_NAME,
        timestamp=datetime.now(UTC) + timedelta(milliseconds=1)
    )

    chats_collection_dep.update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": {
            "lastMessage": bot_response_content,
            "lastActivity": datetime.now(UTC)
        }}
    )

    return MessageSuccessResponse(
        success=True,
        message=ConversationTurn(
            question=bot_message_instance.content,
            answer="",
            timestamp=bot_message_instance.timestamp,
            role="assistant"
        )
    )
@app.get("/api/chats/{chat_id}/messages", response_model=TransformedChatMessagesResponse, responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def get_chat_messages(
    chat_id: str,
    request: Request,
    # current_user_id: str = Depends(get_current_user_id),  # AUTH BYPASS
):
    current_user_id = "test_user_for_bypass"

    chats_collection_dep = request.app.state.chats_collection
    messages_collection_dep = request.app.state.messages_collection

    chat = chats_collection_dep.find_one({"_id": ObjectId(chat_id)})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages_raw = list(messages_collection_dep.find(
        {"chatId": chat_id},
        {"_id": 1, "chatId": 1, "userId": 1, "content": 1, "timestamp": 1, "sender": 1}
    ).sort("timestamp", 1))

    conversation_turns: List[ConversationTurn] = []

    for msg in messages_raw:
        sender_id = str(msg['userId'])
        content = msg['content']
        timestamp = msg.get('timestamp', datetime.now(UTC))
        sender_role_actual = 'assistant' if sender_id == BOT_USER_ID else 'user'

        if sender_role_actual == 'assistant':
            conversation_turns.append(ConversationTurn(
                question=content,
                answer="",
                timestamp=timestamp,
                role="assistant"
            ))
        else:
            conversation_turns.append(ConversationTurn(
                question="",
                answer=content,
                timestamp=timestamp,
                role="user"
            ))

    return TransformedChatMessagesResponse(
        id=chat_id,
        session_uuid=chat_id,
        messages=conversation_turns
    )
@app.get("/api/chat/init", response_model=MessagesListResponse, responses={500: {"model": ErrorResponse}})
async def initialize_chat_session(request: Request):
    messages_collection_dep = request.app.state.messages_collection
    chats_collection_dep = request.app.state.chats_collection

    session_uuid = request.session.get("chat_uuid")
    initial_ai_question = "What is the name or model of the product?"
    first_bot_message_content = f"Hello! {initial_ai_question}"

    current_user_id = "temp_user_id_for_init"

    if not session_uuid:
        session_uuid = str(ObjectId())
        request.session["chat_uuid"] = session_uuid

        llm_store_message(
            messages_collection=messages_collection_dep,
            chat_id=session_uuid,
            user_id=BOT_USER_ID,
            content=first_bot_message_content,
            sender=BOT_SENDER_NAME
        )

        chats_collection_dep.insert_one({
            "_id": ObjectId(session_uuid),
            "participants": [BOT_USER_ID, current_user_id],
            "title": "Initial Chat",
            "createdAt": datetime.now(UTC),
            "lastActivity": datetime.now(UTC),
            "lastMessage": first_bot_message_content
        })

    qa_log_raw = messages_collection_dep.find({"chatId": session_uuid}).sort("timestamp", 1)
    qa_log = [MessageModel(**{**msg, 'id': str(msg['_id'])}) for msg in qa_log_raw]

    return MessagesListResponse(messages=qa_log)
@app.post("/api/trigger_search_pipeline", response_model=SuccessMessageResponse, responses={500: {"model": ErrorResponse}})
async def trigger_search_pipeline_api(request: Request):
    from back_end_llm import run_search_pipeline  # Avoid circular import

    session_uuid = request.session.get("chat_uuid")
    if not session_uuid:
        return JSONResponse(status_code=400, content={"detail": "Session UUID not found."})

    def run_module2_sync(session_id: str):
        try:
            print(f"🔍 Module 2 search pipeline started for session: {session_id}...")
            run_search_pipeline()
            print(f"✅ Module 2 search pipeline finished for session: {session_id}.")
        except Exception as e:
            print(f"❌ Error in Module 2 pipeline for session {session_id}: {e}")

    async def run_module2_background_task(session_id: str):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run_module2_sync, session_id)

    asyncio.create_task(run_module2_background_task(session_uuid))

    return SuccessMessageResponse(message="Search pipeline started in the background.")

