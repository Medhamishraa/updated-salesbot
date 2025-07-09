This Python script scrapes a website to extract visible text and links, then saves the results in a clean JSON format. It supports two scraping methods:
   <li>Selenium – uses a headless browser to load and scrape the page locally.</li>
   <li>Bright Data API – scrapes the page remotely using Bright Data’s collector.</li>

It automatically extracts the company name from the URL and structures the output using the ScrapedData model (via Pydantic).
Finally, it prints the result and saves it as scraped_result.json.

<h3>Classes</h3>
<ol>
<li> ScrapedData (from pydantic.BaseModel)
Represents the structured format for scraped data.
Fields:
<ul>
<li>url (HttpUrl): The source URL.</li>
<li>company_name (str): Inferred name of the company from the domain.</li>
<li>text_content (List[str]): All visible text extracted from the page.</li>
<li>links (List[HttpUrl]): All extracted hyperlinks on the page.</li>
</ul>
</li>
	
<li> CompanyScraper
Main class for scraping website content either with Selenium or Bright Data.
Constructor:
<ul>
CompanyScraper(use_bright_data=False, bright_data_api_key="", bright_collector_id="")<br>
Arguments:
<ul>
<li>use_bright_data (bool): Choose between local browser scraping (Selenium) or remote scraping (Bright Data).</li>
<li>bright_data_api_key (str): API key for Bright Data (required if using Bright Data).</li>
<li>bright_collector_id (str): Collector ID for the Bright Data collector.</li>
</ul>
</li>
</ol>
	
<h3>Methods</h3>
<ol>
<li>extract_company_name(url: str) -> str
Extracts the company name from a given URL (e.g., "https://www.microsoft.com" → "microsoft").

Input:
<ul>
<li>url: Any valid website URL.</li>
<li>Returns:Company name as a lowercase string.</li>
</ul></li>

<li>scrape_with_selenium(url: str) -> ScrapedData
Scrapes the page using a headless Chrome browser (Selenium).

Input:
<ul><li>
url: The URL to scrape.</li>
<li>Returns:

A ScrapedData object containing text and links.</li>
</li></ul></li>

<li>scrape_with_bright_data(url: str) -> ScrapedData
Scrapes the page using Bright Data’s Data Collector API. Waits for results and parses the returned HTML.
Input:
<ul>
<li>url: The URL to scrape.</li>
<li>Returns:
A ScrapedData object with extracted text and hyperlinks.</li></ul>

Note:Enter a valid bright_data_api_key and bright_collector_id.
</li>

<li>scrape(url: str) -> ScrapedData
Wrapper method that calls either Selenium or Bright Data based on the use_bright_data flag.

Input:
<ul>
<li>url: The target website.</li>
<li>Returns:
A ScrapedData instance with structured information.</li></ul>
</li>
</ol>

<h3> Main Execution Block</h3>
if __name__ == "__main__":
What it does:
<li>Sets the URL to scrape (https://www.microsoft.com/en-us/about)</li>
<li>Initializes the scraper (Selenium or Bright Data).</li>
<li>Runs the scraping.</li>
<li>Prints the output in console.</li>
<li>Saves the results as scraped_result.json.</li>

<h3>Output Format (scraped_result.json)</h3>
{
    "url": "https://www.microsoft.com/en-us/about",
    "company_name": "microsoft",
    "text_content": [
        "We empower the world",,
        ...
    ],
    "links": [
        "https://www.microsoft.com/en-us/about",
        "https://careers.microsoft.com",
        ...
    ]
}
