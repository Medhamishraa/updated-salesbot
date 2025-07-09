import asyncio
import httpx
from pydantic import BaseModel
from pydantic_ai import Tool
from dotenv import load_dotenv
from db_utils.db_config import get_collection
from datetime import datetime, timezone

load_dotenv()

class HNScrapeInput(BaseModel):
    company: str

@Tool
async def hn_scrape_tool(input: HNScrapeInput) -> str:
    company = input.company.strip().lower()
    url = f"https://hn.algolia.com/api/v1/search?query={company}&tags=story"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            result = response.json()
    except Exception as e:
        return f"Error fetching data from Hacker News for {company.title()}: {e}"

    posts = result.get("hits", [])
    clean_posts = [
        {
            "id": post.get("objectID", ""),
            "author": post.get("author", ""),
            "url": post.get("url", ""),
            "created_at": post.get("created_at", ""),
            "num_comments": post.get("num_comments", 0),
            "title": post.get("title") or post.get("story_title", "")
        }
        for post in posts
    ]

    collection = get_collection("company", "company_data")
    collection.update_one(
        {"company": company},
        {"$set": {
            "hacker_news": {
                "posts": clean_posts,
                "scraped_at": datetime.now(timezone.utc).isoformat()
            }
        }},
        upsert=True
    )

    return f"Scraped {len(clean_posts)} posts for {company.title()} from Hacker News."

