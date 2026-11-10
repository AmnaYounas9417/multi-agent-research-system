import os

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from tavily import TavilyClient

import config


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information."""
    api_key = config.get_key("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing. Add it to your .env file.")

    result = TavilyClient(api_key=api_key).search(query=query, max_results=5)
    formatted_results = []
    for item in result.get("results", []):
        content = " ".join(item.get("content", "").split())
        snippet = content[:300].rstrip()
        if len(content) > 300:
            snippet += "..."
        formatted_results.append(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Snippet: {snippet}"
        )

    return "\n---------------------------\n".join(formatted_results) or "No results found."


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL."""
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as exc:
        return f"Could not scrape URL: {exc}"


if __name__ == "__main__":
    print(web_search.invoke({"query": "What are the recent news of technology?"}))
    print(scrape_url.invoke({"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"}))
