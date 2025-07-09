# agent.py
import os
from pydantic_ai import Agent

# Import all tools
from youtube_scraper_tool import generate_product_summary
from hacker_news_tool import hn_scrape_tool
from website_scraper_tool import scrape_website_tool_async
from dotenv import load_dotenv
import os

load_dotenv()  # 👈 this loads values from .env into os.environ
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create the unified agent
agent = Agent(
    tools=[
        generate_product_summary,
        hn_scrape_tool,
        scrape_website_tool_async
    ],
    model="gpt-4o",
    system_prompt="You are an intelligent data collector that can scrape YouTube videos, Hacker News posts, and company websites based on user requests."
)

# Optional run block
if __name__ == "__main__":
    import asyncio
    async def main():
        query = input("What would you like to do? ").strip()
        result = await agent.run(query)
        print("\nAgent Result:\n", result.output)
    asyncio.run(main())
