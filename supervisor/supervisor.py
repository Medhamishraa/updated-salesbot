from pydantic_models import PiggyBank
from prompts import system_message
from pydantic_ai import Agent
from utils import analyze_chat_and_scrape
from utils import run_scraper_tool_logic, run_hn_scraper_tool_logic, run_video_processor
import json
from typing import Tuple

# ✅ Define the agent
agent = Agent(
    tools=[run_scraper_tool_logic, run_hn_scraper_tool_logic, run_video_processor],
    system_message=system_message,
    model="gpt-4o",
    max_tool_retries=3
)

def load_inputs(chat_path: str = "chatbot_db.chat_sessions.json", company_path: str = "companies.json") -> Tuple[list, list]:
    with open(chat_path, "r", encoding="utf-8") as chat_file:
        chat_data = json.load(chat_file)

    with open(company_path, "r", encoding="utf-8") as company_file:
        company_data = json.load(company_file)

    return chat_data, company_data


async def smart_scrape_companies() -> PiggyBank:
    try:
        # 🔍 Load input files
        chat_data, company_data = load_inputs()

        # 🔁 Pass inputs to the core analysis function
        results = await analyze_chat_and_scrape(chat_data=chat_data, company_data=company_data)

        return PiggyBank(companies=results)

    except FileNotFoundError as fnf:
        raise RuntimeError(f"Missing file: {fnf}")
    except Exception as e:
        raise RuntimeError(f"Internal Error: {str(e)}")
