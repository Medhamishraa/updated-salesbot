# website_scraper_tool.py
import os
import json
import time
import asyncio
from typing import List
from urllib.parse import urlparse, urljoin
from collections import deque
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pydantic import BaseModel, HttpUrl
from pydantic_ai import Tool

load_dotenv()

# ----------------------- Models -----------------------

class WebsiteContent(BaseModel):
    url: str
    company_name: str
    text_content: List[str]
    links: List[str]

class ScraperInput(BaseModel):
    url: HttpUrl
    max_pages: int = 0  # 0 = no limit

class ScraperOutput(BaseModel):
    json_file: str
    summary_file: str
    summary_text: str

class SummaryInput(BaseModel):
    full_text: str

class SummaryOutput(BaseModel):
    summary: str

# ----------------------- Summary Tool -----------------------

# @Tool
def summarize_with_pydantic_ai(input: SummaryInput) -> SummaryOutput:
    """Summarizes extracted company website content into a paragraph."""
    content = input.full_text[:12000]
    prompt = f"""
You are a company analyst. Summarize the following extracted content from a company's website.

Mention:
- What the company does
- Its services or products
- Its values/vision/mission (if available)
- Its market focus or target users

Content:
{content}
"""
    return SummaryOutput(summary=prompt)

# ----------------------- Scraper -----------------------

class CompanyWebsiteScraper:
    def normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed._replace(query="", fragment="").geturl().rstrip("/")

    def _extract_domain_as_company(self, url: str) -> str:
        hostname = urlparse(url).hostname or ""
        parts = hostname.replace("www.", "").split(".")
        return parts[0] if parts else "unknown"

    def _scrape_using_selenium(self, url: str) -> WebsiteContent:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=Service(), options=options)
        driver.get(url)

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        try:
            container = driver.find_element(By.TAG_NAME, "main")
        except:
            container = driver.find_element(By.TAG_NAME, "body")

        all_elements = container.find_elements(By.XPATH, ".//*")
        visible_texts = []
        seen_texts = set()

        for elem in all_elements:
            try:
                if elem.is_displayed():
                    if not elem.find_elements(By.XPATH, "./*"):
                        text = elem.text.strip()
                        if text and text not in seen_texts:
                            visible_texts.append(text)
                            seen_texts.add(text)
            except:
                continue

        links = {
            a.get_attribute('href')
            for a in driver.find_elements(By.TAG_NAME, 'a')
            if a.get_attribute('href') and a.get_attribute('href').startswith("http")
        }

        driver.quit()

        return WebsiteContent(
            url=url,
            company_name=self._extract_domain_as_company(url),
            text_content=visible_texts,
            links=list(links)
        )

    def extract_website_content(self, url: str) -> WebsiteContent:
        return self._scrape_using_selenium(url)

    

    def crawl_website(self, base_url: str, max_pages: int = 0) -> List[WebsiteContent]:
        base_url = self.normalize_url(base_url)
        base_pattern = re.compile(rf"^{re.escape(base_url)}(/.*)?$")

        visited = set()
        to_visit = deque([base_url])
        scraped_data = []

        while to_visit:
            current_url = to_visit.popleft()
            norm_url = self.normalize_url(current_url)
            if norm_url in visited:
                continue

            try:
                print(f"Scraping: {current_url}")
                content = self.extract_website_content(current_url)
                scraped_data.append(content)
                visited.add(norm_url)

                for link in content.links:
                    full_url = self.normalize_url(
                        link if urlparse(link).netloc else urljoin(base_url, link)
                    )

                    if full_url not in visited and base_pattern.match(full_url):
                        to_visit.append(full_url)

                if max_pages > 0 and len(visited) >= max_pages:
                    break

            except Exception as e:
                print(f"Error scraping {current_url}: {e}")
                continue

        return scraped_data


    def save_all_to_json(self, data_list: List[WebsiteContent], path: str = "full_scrape_output.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([json.loads(d.model_dump_json()) for d in data_list], f, ensure_ascii=False, indent=4)

# ----------------------- Main Logic -----------------------

async def run_scraper_tool_logic(input_data: ScraperInput) -> ScraperOutput:
    scraper = CompanyWebsiteScraper()
    results = scraper.crawl_website(str(input_data.url), max_pages=input_data.max_pages)
    scraper.save_all_to_json(results, "full_scrape_output.json")

    combined_text = "\n".join(" ".join(r.text_content) for r in results)
    summary_result = summarize_with_pydantic_ai(SummaryInput(full_text=combined_text))
    summary = summary_result.summary

    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, ensure_ascii=False, indent=4)

    return ScraperOutput(
        json_file="full_scrape_output.json",
        summary_file="summary.json",
        summary_text=summary
    )

# ----------------------- Tool Wrapper -----------------------

@Tool
async def scrape_company_website(input_data: ScraperInput) -> ScraperOutput:
    """Scrapes full content of a website and summarizes it without DB dependencies."""
    return await asyncio.to_thread(lambda: asyncio.run(run_scraper_tool_logic(input_data)))

# ----------------------- Test Usage -----------------------

# if __name__ == "__main__":
#     asyncio.run(run_scraper_tool_logic(ScraperInput(url="https://www.yugen.ai/", max_pages=0)))
