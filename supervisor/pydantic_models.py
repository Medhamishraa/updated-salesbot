from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class UserChatHistory(BaseModel):
    session_id: str
    history: List[ChatMessage]


class Company(BaseModel):
    name: str
    website: HttpUrl


class AgentInput(BaseModel):
    chat_data: UserChatHistory
    companies: List[Company]


class WebsiteScraperData(BaseModel):
    text_content: List[str]
    links: List[HttpUrl]


class CompanyScrapedData(BaseModel):
    name: str
    website_scraper: WebsiteScraperData
    hn_articles: List[str]
    yt_scraper: str


class PiggyBank(BaseModel):
    companies: List[CompanyScrapedData]