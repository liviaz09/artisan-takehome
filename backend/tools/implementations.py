"""
tools/implementations.py — Tool execution functions.

These are the Python functions that run when Claude calls a tool.
Each function calls Firecrawl, pipes the result through the chunker,
and returns only the relevant snippets — never raw markdown.

The agent never sees full page content. It only ever sees
pre-filtered, goal-relevant snippets as tool observations.
This enforces the snippet discipline at the tool boundary.
"""

import os
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from chunker import get_relevant_snippets

load_dotenv()

_firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


def scrape_page(url: str, goal: str) -> str:
    """
    Fetch a page via Firecrawl, extract relevant snippets for the goal.
    Returns a formatted string of snippets for the agent's observation.
    """
    try:
        result = _firecrawl.scrape_url(url, params={"formats": ["markdown"]})

        if isinstance(result, dict):
            markdown = result.get("markdown", "")
        else:
            markdown = getattr(result, "markdown", "") or ""

        if not markdown.strip():
            return f"No content found at {url}"

        snippets = get_relevant_snippets(markdown, url, goal)

        if not snippets:
            return f"Page fetched but no content relevant to '{goal}' found at {url}"

        return _format_snippets(snippets)

    except Exception as e:
        return f"Failed to scrape {url}: {str(e)}"


def search_web(query: str, goal: str) -> str:
    """
    Search the web via Firecrawl, extract relevant snippets from results.
    Returns a formatted string of snippets for the agent's observation.
    """
    try:
        result = _firecrawl.search(query, params={"limit": 5})
        items  = result if isinstance(result, list) else getattr(result, "data", []) or []

        all_snippets = []
        for item in items:
            if isinstance(item, dict):
                url      = item.get("url", "")
                markdown = item.get("markdown", "") or item.get("content", "")
            else:
                url      = getattr(item, "url", "")
                markdown = getattr(item, "markdown", "") or getattr(item, "content", "")

            if markdown and url:
                snippets = get_relevant_snippets(markdown, url, goal)
                all_snippets.extend(snippets)

        if not all_snippets:
            return f"No relevant results found for query: '{query}'"

        # Re-rank across all search results and take top-k
        all_snippets.sort(key=lambda x: x["score"], reverse=True)
        return _format_snippets(all_snippets[:8])

    except Exception as e:
        return f"Search failed for '{query}': {str(e)}"


def _format_snippets(snippets: list[dict]) -> str:
    """
    Format snippets as a readable string for the agent's observation.
    Each snippet is tagged with its source URL for claim traceability.
    """
    parts = []
    for i, s in enumerate(snippets, 1):
        parts.append(f"[{i}] SOURCE: {s['url']}\n{s['text']}")
    return "\n\n".join(parts)
