"""
tools/implementations.py — Tool execution functions.

Uses Jina AI Reader for scraping and DuckDuckGo for web search.
Both are free, require no API keys, and have no quota limits.

Jina AI Reader (r.jina.ai):
  - Prefix any URL with https://r.jina.ai/ to get clean markdown
  - Handles JS rendering server-side
  - No API key, no quota, completely free

DuckDuckGo HTML search:
  - Simple HTTP request to html.duckduckgo.com
  - Returns search result snippets
  - No API key required

The agent never sees full page content — only top-k relevant snippets
filtered by cosine similarity inside each tool.
"""

import httpx
from chunker import get_relevant_snippets

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_page(url: str, goal: str) -> str:
    """
    Fetch a page via Jina AI Reader and return relevant snippets for the goal.
    Jina renders JS and returns clean markdown — no API key needed.
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"
        response = httpx.get(
            jina_url,
            headers={**_HEADERS, "Accept": "text/markdown"},
            timeout=30,
            follow_redirects=True,
        )
        response.raise_for_status()
        markdown = response.text

        if not markdown.strip():
            return f"No content found at {url}"

        print(f"[implementations] scrape_page: {url[:60]} — {len(markdown)} chars")

        snippets = get_relevant_snippets(markdown, url, goal)
        if not snippets:
            return f"Page fetched but no content relevant to '{goal}' found at {url}"

        return _format_snippets(snippets)

    except Exception as e:
        return f"Failed to scrape {url}: {str(e)}"


def search_web(query: str, goal: str) -> str:
    """
    Search the web via DuckDuckGo and return relevant snippets.
    For each result, fetches the page via Jina for full content.
    No API key required.
    """
    try:
        # Get search results from DuckDuckGo HTML interface
        from bs4 import BeautifulSoup
        from urllib.parse import quote

        ddg_url  = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        response = httpx.get(ddg_url, headers=_HEADERS, timeout=15, follow_redirects=True)
        soup     = BeautifulSoup(response.text, "html.parser")

        # Extract result URLs and snippets
        all_snippets = []
        for result in soup.select(".result")[:5]:
            snippet_el = result.select_one(".result__snippet")
            link_el    = result.select_one("a.result__url")

            if not snippet_el:
                continue

            # Get URL from the result link
            href = ""
            if link_el and link_el.get("href"):
                href = link_el["href"]
                if href.startswith("//"):
                    href = "https:" + href
                elif not href.startswith("http"):
                    href = "https://" + href

            text = snippet_el.get_text(strip=True)
            if text and href:
                all_snippets.append({"text": text, "url": href, "score": 0.5})

        if not all_snippets:
            return f"No results found for query: '{query}'"

        # Score snippets against goal and return top results
        from chunker import get_relevant_snippets as grs
        combined_text = "\n\n".join(s["text"] for s in all_snippets)
        scored = grs(combined_text, query, goal)

        if scored:
            # Re-attach URLs to scored snippets
            for snippet in scored:
                snippet["url"] = next(
                    (s["url"] for s in all_snippets if s["text"][:50] in snippet["text"]),
                    query
                )
            return _format_snippets(scored)

        return _format_snippets(all_snippets[:5])

    except Exception as e:
        return f"Search failed for '{query}': {str(e)}"


def _format_snippets(snippets: list[dict]) -> str:
    """
    Format snippets as a readable string for the agent's observation.
    Each snippet tagged with its source URL for claim traceability.
    """
    parts = []
    for i, s in enumerate(snippets, 1):
        parts.append(f"[{i}] SOURCE: {s['url']}\n{s['text']}")
    return "\n\n".join(parts)
