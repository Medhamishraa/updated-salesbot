import os
import json
import re
import asyncio
from typing import List, Dict
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import httpx
from youtubesearchpython import VideosSearch
import yt_dlp
from huggingface_hub import InferenceClient
from openai import OpenAI
from pydantic_models import CompanyScrapedData, PiggyBank
from access import WebsiteExtractor
from pydantic_ai import Agent
from prompts import system_message

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 3000

class ScraperInput(BaseModel):
    name: str
    website: HttpUrl

class ScraperOutput(BaseModel):
    text_content: List[str]
    links: List[HttpUrl]

class HNScrapeInput(BaseModel):
    company: str

class ProductInput(BaseModel):
    product_name: str

async def run_scraper_tool_logic(input_data: ScraperInput) -> ScraperOutput:
    extractor = WebsiteExtractor()
    result = extractor.extract(input_data.website)
    return ScraperOutput(
        text_content=result.text_content,
        links=result.links
    )

async def run_hn_scraper_tool_logic(input_data: HNScrapeInput) -> List[str]:
    try:
        url = f"https://hn.algolia.com/api/v1/search?query={input_data.company.strip()}&tags=story"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            result = response.json()
        posts = result.get("hits", [])
        return [f"{post.get('title', '')} - {post.get('url', '')}" for post in posts if post.get("title")]
    except Exception as e:
        return [f"HN error: {e}"]

class YTDLogger:
    def __init__(self, log_file): self.log_file = log_file
    def debug(self, msg): self._write(msg)
    def warning(self, msg): self._write(msg)
    def error(self, msg): self._write(msg)
    def _write(self, msg): open(self.log_file, 'a', encoding='utf-8').write(msg + '\n')

class VideoProcessor:
    def __init__(self, hf_token, hf_model):
        self.client = InferenceClient(model=hf_model, token=hf_token)

    @staticmethod
    def sanitize(name): return re.sub(r'[\\/*?:"<>|]', "", name).strip()

    @staticmethod
    def extract_text_from_vtt(vtt):
        lines = vtt.splitlines()
        return '\n'.join(
            re.sub(r'<[^>]+>', '', line).strip()
            for line in lines
            if line.strip() and '-->' not in line and not line.startswith(('WEBVTT', 'Kind:', 'Language:')) and '[Music]' not in line
        )

    def clean_text(self, raw_text):
        prompt = (
            "Clean this text by removing repeated or overlapping phrases. Keep only meaningful content:\n\n"
            f"{raw_text}"
        )
        try:
            return self.client.text_generation(prompt, max_new_tokens=2048).strip()
        except Exception as e:
            return raw_text

    def get_video_urls(self, query, max_results=10):
        results = VideosSearch(f"{query} demo OR review", limit=max_results).result()['result']
        return [r['link'] for r in results]

    async def download_and_clean(self, video_url, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            title = self.sanitize(info.get('title', 'video'))
            base_path = os.path.join(output_dir, title)
            vtt_file = f"{base_path}.en.vtt"
            json_file = f"{base_path}.json"

        if os.path.exists(json_file):
            return

        ydl_opts = {
            'quiet': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'skip_download': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'vtt',
            'outtmpl': base_path + '.%(ext)s',
            'logger': YTDLogger(base_path + ".log")
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if os.path.exists(vtt_file):
            with open(vtt_file, 'r', encoding='utf-8') as f:
                raw = f.read()
            loop = asyncio.get_running_loop()
            cleaned = await loop.run_in_executor(None, self.clean_text, self.extract_text_from_vtt(raw))
        else:
            cleaned = ""

        with open(json_file, 'w', encoding='utf-8') as jf:
            json.dump({
                "video_title": title,
                "video_url": video_url,
                "transcript": cleaned
            }, jf, indent=2, ensure_ascii=False)

class TranscriptSummarizer:
    def __init__(self, transcript_dir, product_name):
        self.dir = transcript_dir
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.product = product_name.lower()

    def _load_transcripts(self):
        texts = []
        for f in os.listdir(self.dir):
            if f.endswith(".json"):
                with open(os.path.join(self.dir, f), 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    transcript = data.get("transcript", "")
                    if transcript.strip():
                        texts.append(transcript.strip())
        return "\n\n".join(texts)

    def _chunk_text(self, text):
        return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    async def _summarize_chunk(self, text):
        prompt = (
            "Summarize the following transcript **only focusing on the target product** and its "
            "capabilities, use cases, features, and benefits. "
            "Do not elaborate on comparisons or other tools:\n\n"
            f"{text}"
        )
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a product analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    async def summarize(self):
        text = self._load_transcripts()
        if not text:
            return {"summary": "No usable transcripts found."}

        chunks = self._chunk_text(text)
        partials = await asyncio.gather(*[self._summarize_chunk(chunk) for chunk in chunks if chunk.strip()])
        combined = "\n\n".join(partials)
        final = await self._summarize_chunk(combined)

        output_path = os.path.join(self.dir, "combined_summary.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "product_name": self.product,
                "summary": final
            }, f, indent=2, ensure_ascii=False)
        return {"transcripts_cleaned": True, "summary": final}

async def run_video_processor(product_name: str):
    processor = VideoProcessor(HF_TOKEN, HF_MODEL)
    base_dir = f"{VideoProcessor.sanitize(product_name)}_captions"
    os.makedirs(base_dir, exist_ok=True)

    urls = processor.get_video_urls(product_name, max_results=10)
    await asyncio.gather(*(processor.download_and_clean(url, base_dir) for url in urls))

    summarizer = TranscriptSummarizer(transcript_dir=base_dir, product_name=product_name)
    return await summarizer.summarize()

async def get_matching_companies_from_chat(chat_data, company_data, agent) -> List[Dict]:
    import json
 
    session = chat_data[0]
    messages = session.get("messages", [])
    if not messages:
        raise ValueError("❌ No messages found in chat_data[0]['messages']. Please check JSON format.")

    chat_text = ""
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            chat_text += f"User: {msg.get('answer', '').strip()}\n"
        elif role == "assistant":
            chat_text += f"Assistant: {msg.get('question', '').strip()}\n"

    all_companies = company_data[0].get("companies", [])
    formatted = [f"{c['name']} - {c.get('website', '')}" for c in all_companies]

    prompt = (
        f"User chat history:\n{chat_text}\n\n"
        f"Companies:\n{chr(10).join(formatted)}\n\n"
        "Return a list of the most relevant companies in this format:\n"
        "[{\"name\": \"CompanyName\", \"website\": \"https://...\"}, ...]"
    )

    response = await agent.run(prompt)
    print("🧾 Raw result:", response.output)
    return json.loads(response.content)

async def analyze_chat_and_scrape(chat_data, company_data) -> List[CompanyScrapedData]:

    agent = Agent(
    system_message=system_message,
    tools=[],
    model="gpt-4o",
    backend="openai",
    output_model=PiggyBank
)
    selected = await get_matching_companies_from_chat(chat_data, company_data, agent)

    results = []
    for company in selected:
        name = company["name"]
        website = company["website"]

        website_data = await run_scraper_tool_logic(ScraperInput(name=name, website=website))
        hn_data = await run_hn_scraper_tool_logic(HNScrapeInput(company=name))
        yt_data = await run_video_processor(name)

        results.append(CompanyScrapedData(
            name=name,
            website_scraper=website_data,
            hn_articles=hn_data,
            yt_scraper=yt_data["summary"]
        ))

    return results
