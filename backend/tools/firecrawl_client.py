"""
firecrawl_client.py — Live web fetching tool.

Design rationale:
  Firecrawl gives us JS-rendered, markdown-converted content in one API call.
  This means our chunker receives clean prose rather than HTML noise —
  directly reducing token waste and improving extraction quality.

  We use three Firecrawl capabilities:
    /map    → discover all URLs on a site (feeds our planning step)
    /scrape → fetch a single URL as clean markdown
    /search → web-wide search for signals about a company

  Token discipline: we never pass raw full-page markdown to the LLM.
  All content goes through chunker.py first.
"""

import os
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()

_client = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Pages we always want to prioritize for sender analysis
SENDER_PAGE_PRIORITIES = [
    "",          # homepage
    "/about",
    "/customers",
    "/case-studies",
    "/pricing",
    "/product",
    "/solutions",
    "/why-us",
    "/blog",
]

# Pages we always want to prioritize for target analysis
TARGET_PAGE_PRIORITIES = [
    "",          # homepage
    "/about",
    "/team",
    "/careers",
    "/blog",
    "/press",
    "/news",
    "/product",
    "/solutions",
]


def map_site(url: str, limit: int = 20) -> list[str]:
    """
    Discover URLs on a site using Firecrawl's /map endpoint.
    Returns a ranked list: priority pages first, then discovered pages.
    limit: max pages to consider (cost control).
    """
    try:
        result = _client.map_url(url, params={"limit": limit})
        # firecrawl-py returns either a list or an object with .links
        if isinstance(result, list):
            discovered = result
        else:
            discovered = getattr(result, "links", []) or []

        base = url.rstrip("/")
        discovered_set = set(discovered)

        # Build priority-first list
        ordered = []
        for path in SENDER_PAGE_PRIORITIES:
            candidate = base + path
            if candidate in discovered_set or path == "":
                ordered.append(candidate if path else base)

        # Append remaining discovered URLs up to limit
        for link in discovered:
            if link not in ordered:
                ordered.append(link)

        return ordered[:limit]
    except Exception as e:
        print(f"[firecrawl] map_site error for {url}: {e}")
        return [url]


def scrape_page(url: str) -> dict | None:
    """
    Scrape a single URL. Returns:
      { "url": str, "markdown": str, "metadata": dict }
    Returns None on failure (don't crash the agent on a 404).
    """
    try:
        result = _client.scrape_url(url, params={"formats": ["markdown"]})
        if not result:
            return None

        # firecrawl-py can return an object or dict
        if isinstance(result, dict):
            markdown = result.get("markdown", "")
            metadata = result.get("metadata", {})
        else:
            markdown = getattr(result, "markdown", "") or ""
            metadata = getattr(result, "metadata", {}) or {}

        if not markdown.strip():
            return None

        return {"url": url, "markdown": markdown, "metadata": metadata}
    except Exception as e:
        print(f"[firecrawl] scrape_page error for {url}: {e}")
        return None


def scrape_pages(urls: list[str]) -> list[dict]:
    """
    Scrape multiple URLs. Skips failures silently.
    Returns list of { url, markdown, metadata } dicts.
    """
    results = []
    for url in urls:
        page = scrape_page(url)
        if page:
            results.append(page)
    return results


def search_web(query: str, limit: int = 5) -> list[dict]:
    """
    Web-wide search via Firecrawl's /search endpoint.
    Used to find signals about a target company that aren't on their site:
    funding news, press coverage, recent hires, competitor mentions.
    Returns list of { url, markdown, title } dicts.
    """
    try:
        result = _client.search(query, params={"limit": limit})

        if isinstance(result, list):
            items = result
        else:
            items = getattr(result, "data", []) or []

        pages = []
        for item in items:
            if isinstance(item, dict):
                pages.append({
                    "url": item.get("url", ""),
                    "markdown": item.get("markdown", "") or item.get("content", ""),
                    "title": item.get("metadata", {}).get("title", "") if isinstance(item.get("metadata"), dict) else "",
                })
            else:
                pages.append({
                    "url": getattr(item, "url", ""),
                    "markdown": getattr(item, "markdown", "") or getattr(item, "content", ""),
                    "title": getattr(getattr(item, "metadata", None), "title", "") if hasattr(item, "metadata") else "",
                })
        return [p for p in pages if p["markdown"]]
    except Exception as e:
        print(f"[firecrawl] search_web error for '{query}': {e}")
        return []
