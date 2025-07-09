from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re


class WebsiteContent(BaseModel):
    url: Optional[HttpUrl]
    company_name: str
    text_content: List[str]
    links: List[HttpUrl]


def extract_body_content(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    body = soup.body or soup
    return str(body)


def clean_body_content(body_content: str) -> str:
    soup = BeautifulSoup(body_content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.extract()
    text = soup.get_text(separator="\n")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


class WebsiteExtractor:
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")

    def _extract_domain_as_company(self, url: str) -> str:
        hostname = urlparse(url).hostname or ""
        parts = hostname.replace("www.", "").split(".")
        return parts[0] if parts else "unknown"

    def extract(self, url: str) -> WebsiteContent:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        driver.get(url)
        print(f"\U0001F30D Loaded: {url}")

        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html = driver.page_source
        driver.quit()

        body = extract_body_content(html)
        cleaned = clean_body_content(body)
        text_blocks = cleaned.split("\n")

        soup = BeautifulSoup(html, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith("http")]

        return WebsiteContent(
            url=url,
            company_name=self._extract_domain_as_company(url),
            text_content=text_blocks,
            links=links
        )
