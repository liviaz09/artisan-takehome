"""
sender_agent.py — Mode 1: ICP and value proposition generation.

Agent loop:
  1. CHECK CACHE   → return immediately if we've analyzed this sender before
  2. MAP + SELECT  → discover URLs, filter deterministically (no LLM)
  3. FETCH         → Firecrawl scrapes selected pages
  4. CHUNK+EMBED   → RecursiveCharacterTextSplitter + text-embedding-3-small
  5. SYNTHESIZE    → Claude extracts value prop + ICP from top-k snippets only
  6. CACHE WRITE   → save results for future runs

What we removed vs v1:
  - The Claude planning call (_plan_fetch_targets) is gone entirely.
    We already know which pages matter for sender ICP research — homepage,
    about, customers, pricing, case-studies. Hardcoding this is faster,
    cheaper, and more reliable than asking Claude to decide.

  - Research goals are hardcoded strings passed directly to the embedding
    model. Claude doesn't decide what to look for — we do.
"""

import os
import anthropic
from dotenv import load_dotenv

from tools.cache import cache_get, cache_set, cache_set_pages, cache_get_pages
from tools.firecrawl_client import map_site, scrape_pages, select_pages
from tools.chunker import extract_relevant_snippets, format_snippets_for_prompt
from prompts.prompts import ICP_EXTRACTION_PROMPT

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Hardcoded research goal for sender ICP extraction.
# We know exactly what we're looking for — no LLM needed to decide this.
SENDER_RESEARCH_GOAL = (
    "value proposition, target customer profile, ideal customer industries, "
    "company size and segment, pain points solved, customer testimonials, "
    "case studies, differentiators from competitors, pricing tiers and buyer personas"
)


def _synthesize_icp(url: str, snippets: list[dict]) -> dict:
    """
    Synthesis: Claude extracts value prop + ICP from snippets only.
    This is the only LLM call in Mode 1.
    """
    formatted = format_snippets_for_prompt(snippets)
    prompt = ICP_EXTRACTION_PROMPT.format(url=url, snippets=formatted)

    response = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        import json
        return json.loads(raw)
    except Exception as e:
        print(f"[sender_agent] ICP parse error: {e}\nRaw: {raw[:300]}")
        return {
            "value_proposition": "Could not extract — check evidence quality.",
            "icp": {},
            "confidence": "low",
            "evidence_gaps": ["JSON parse failed"]
        }


async def analyze_sender(url: str) -> dict:
    """
    Main entry point for Mode 1.
    Returns: { value_proposition, icp, snippets_used, pages_fetched, cached }
    """
    if not url.startswith("http"):
        url = "https://" + url
    base = url.rstrip("/")

    # ── Step 1: Cache check ──────────────────────────────────────────────────
    cached = cache_get(url)
    if cached and cached.get("value_prop") and cached.get("icp"):
        print(f"[sender_agent] Cache hit for {url}")
        return {
            "value_proposition": cached["value_prop"],
            "icp": cached["icp"],
            "snippets_used": cached.get("snippets_used", []),
            "pages_fetched": list(cached.get("pages", {}).keys()),
            "cached": True,
        }

    print(f"[sender_agent] Cache miss — starting agent loop for {url}")

    # ── Step 2: Map + deterministic page selection ────────────────────────────
    # No LLM call — select_pages uses hardcoded priority paths + URL blocklist
    discovered = map_site(url, limit=30)
    pages_to_fetch = select_pages(discovered, base, mode="sender", limit=6)
    print(f"[sender_agent] Selected {len(pages_to_fetch)} pages: {pages_to_fetch}")

    # ── Step 3: Fetch ─────────────────────────────────────────────────────────
    cached_pages = cache_get_pages(url) or {}
    pages_to_scrape = [p for p in pages_to_fetch if p not in cached_pages]

    scraped = scrape_pages(pages_to_scrape)
    print(f"[sender_agent] Fetched {len(scraped)} new pages")

    all_pages = {**cached_pages}
    for page in scraped:
        all_pages[page["url"]] = page["markdown"]
    cache_set_pages(url, all_pages)

    page_objects = [{"url": u, "markdown": m} for u, m in all_pages.items()]

    # ── Step 4: Chunk + embed + similarity ranking ────────────────────────────
    # Embedding model (not LLM) handles relevance scoring
    snippets = extract_relevant_snippets(page_objects, SENDER_RESEARCH_GOAL, top_k=12)
    print(f"[sender_agent] Selected {len(snippets)} relevant snippets via embeddings")

    # ── Step 5: Synthesize (one Claude call) ─────────────────────────────────
    result = _synthesize_icp(url, snippets)

    # ── Step 6: Cache write ───────────────────────────────────────────────────
    cache_set(url, {
        "type": "sender",
        "value_prop": result.get("value_proposition"),
        "icp": result.get("icp"),
        "snippets_used": snippets,
    })

    return {
        "value_proposition": result.get("value_proposition"),
        "icp": result.get("icp"),
        "confidence": result.get("confidence"),
        "evidence_gaps": result.get("evidence_gaps", []),
        "snippets_used": snippets,
        "pages_fetched": list(all_pages.keys()),
        "cached": False,
    }
