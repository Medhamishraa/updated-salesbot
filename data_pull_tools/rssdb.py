import feedparser
from urllib.parse import quote_plus
from pydantic import BaseModel
from typing import List
from datetime import datetime
from dateutil import parser as date_parser
from html.parser import HTMLParser
from pymongo import MongoClient
import json

# MongoDB URI
MONGO_URI = "mongodb+srv://ayushsinghbasera:YEJTg3zhMwXJcTXm@cluster0.fmzrdga.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Utility class to strip HTML
class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

def strip_html(html: str) -> str:
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_data().strip()

# News Article model
class NewsArticle(BaseModel):
    title: str
    link: str
    published: datetime
    summary: str

# Fetcher class
class GoogleNewsFetcher:
    def __init__(self, search_term: str, max_results: int = 10):
        self.search_term = search_term
        self.max_results = max_results
        self.articles: List[NewsArticle] = []

    def fetch_news(self):
        encoded_term = quote_plus(self.search_term)
        rss_url = f"https://news.google.com/rss/search?pz=1&cf=all&q={encoded_term}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)

        for entry in feed.entries[:self.max_results]:
            published_dt = date_parser.parse(entry.published) if 'published' in entry else None
            clean_summary = strip_html(entry.summary) if 'summary' in entry else ''
            article = NewsArticle(
                title=entry.title,
                link=entry.link,
                published=published_dt,
                summary=clean_summary
            )
            self.articles.append(article)

    def to_json_file(self, filename="news_output.json"):
        if not self.articles:
            self.fetch_news()

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                [article.model_dump(mode="json") for article in self.articles],
                f, ensure_ascii=False, indent=4, default=str
            )

    def save_to_mongodb(self, mongo_uri: str, db_name: str, collection_name: str):
        if not self.articles:
            self.fetch_news()

        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        documents = [article.model_dump(mode="python") for article in self.articles]
        result = collection.insert_many(documents)
        print(f" Inserted {len(result.inserted_ids)} documents into MongoDB.")


if __name__ == "__main__":
    search_term = input("Enter the news topic you want to search for: ").strip()
    
    fetcher = GoogleNewsFetcher(search_term, max_results=10)
    
    # Save to JSON file
    filename = f"{search_term.replace(' ', '_')}.json"
    fetcher.to_json_file(filename)
    
    # Save to MongoDB
    fetcher.save_to_mongodb(
        mongo_uri=MONGO_URI,
        db_name="newsDB",
        collection_name="articlesrss"
    )
