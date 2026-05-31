"""
firecrawl_client.py — Live web fetching tool.

Design rationale:
  Firecrawl gives us JS-rendered, markdown-converted content in one API call.
  This means our chunker receives clean prose rather than HTML noise —
  directly reducing token waste and improving extraction quality.

  Page selection is deterministic — we hardcode which paths matter for each
  mode and use a URL blocklist to filter noise. No LLM call needed here;
  we already know what pages are useful for ICP research vs target research.

  We use three Firecrawl capabilities:
    /map    → discover all URLs on a site
    /scrape → fetch a single URL as clean markdown
    /search → web-wide search for signals about a company
"""

import os
import re
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()

_client = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

# ---------------------------------------------------------------------------
# Hardcoded priority paths — deterministic, no LLM needed
# We know exactly what pages carry ICP signal for a sender company.
# ---------------------------------------------------------------------------
SENDER_PRIORITY_PATHS = [
    "",             # homepage
    "/about",
    "/about-us",
    "/customers",
    "/case-studies",
    "/pricing",
    "/product",
    "/solutions",
    "/why-us",
    "/platform",
    "/features",
]

TARGET_PRIORITY_PATHS = [
    "",             # homepage
    "/about",
    "/about-us",
    "/team",
    "/careers",
    "/blog",
    "/press",
    "/news",
    "/investors",
    "/product",
    "/solutions",
    "/platform",
]

# ---------------------------------------------------------------------------
# Blocklist — URL path segments that never contain useful ICP signal.
# Deterministic grep filter; zero LLM tokens spent on this decision.
# ---------------------------------------------------------------------------
BLOCKED_PATH_PATTERNS = [
    r"/contact",
    r"/support",
    r"/help",
    r"/faq",
    r"/legal",
    r"/privacy",
    r"/terms",
    r"/cookie",
    r"/gdpr",
    r"/sitemap",
    r"/login",
    r"/signup",
    r"/sign-up",
    r"/register",
    r"/demo",
    r"/webinar",
    r"/event",
    r"/404",
    r"/cdn-cgi",
    r"\.(pdf|png|jpg|jpeg|gif|svg|ico|css|js|xml|json)$",
    r"/#",          # anchor-only links
    r"/tag/",
    r"/category/",
    r"/author/",
    r"/page/\d+",
]

_blocked_re = re.compile("|".join(BLOCKED_PATH_PATTERNS), re.IGNORECASE)


def _is_blocked(url: str) -> bool:
    """Return True if this URL should be skipped — deterministic, no LLM."""
    try:
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        path = "/" + path
    except Exception:
        path = url
    return bool(_blocked_re.search(path))


def _priority_sort(urls: list[str], base: str, priority_paths: list[str]) -> list[str]:
    """
    Sort URLs so hardcoded priority pages come first.
    Remaining URLs appended in discovered order after filtering.
    """
    priority_urls = []
    seen = set()

    for path in priority_paths:
        candidate = (base + path).rstrip("/") or base
        if candidate not in seen:
            priority_urls.append(candidate)
            seen.add(candidate)

    remainder = [u for u in urls if u not in seen and not _is_blocked(u)]
    return priority_urls + remainder


def select_pages(discovered: list[str], base: str, mode: str = "sender", limit: int = 6) -> list[str]:
    """
    Deterministically select which pages to fetch.
    Filters blocked URLs, prioritizes known-useful paths, caps at limit.
    No LLM call — we already know what pages matter.
    """
    paths = SENDER_PRIORITY_PATHS if mode == "sender" else TARGET_PRIORITY_PATHS
    ordered = _priority_sort(discovered, base, paths)
    return ordered[:limit]


def map_site(url: str, limit: int = 30) -> list[str]:
    """
    Discover URLs on a site using Firecrawl's /map endpoint.
    Returns raw discovered URLs (filtering happens in select_pages).
    """
    try:
        result = _client.map_url(url, params={"limit": limit})
        if isinstance(result, list):
            discovered = result
        else:
            discovered = getattr(result, "links", []) or []
        return [u for u in discovered if not _is_blocked(u)]
    except Exception as e:
        print(f"[firecrawl] map_site error for {url}: {e}")
        return [url]


def scrape_page(url: str) -> dict | None:
    """
    Scrape a single URL. Returns { url, markdown, metadata } or None on failure.
    """
    try:
        result = _client.scrape_url(url, params={"formats": ["markdown"]})
        if not result:
            return None

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
    """Scrape multiple URLs, skip failures silently."""
    results = []
    for url in urls:
        page = scrape_page(url)
        if page:
            results.append(page)
    return results


def search_web(query: str, limit: int = 5) -> list[dict]:
    """
    Web-wide search via Firecrawl's /search endpoint.
    Used for external signals not on the target's own site:
    funding rounds, press coverage, recent hires, competitor mentions.
    """
    try:
        result = _client.search(query, params={"limit": limit})
        items = result if isinstance(result, list) else getattr(result, "data", []) or []

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
                    "title": "",
                })
        return [p for p in pages if p["markdown"]]
    except Exception as e:
        print(f"[firecrawl] search_web error for '{query}': {e}")
        return []
