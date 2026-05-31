"""
sender_agent.py — Mode 1: ICP and value proposition generation.

Agent loop:
  1. CHECK CACHE   → return immediately if we've analyzed this sender before
  2. MAP SITE      → discover URLs via Firecrawl /map
  3. PLAN          → Claude decides which pages are most valuable
  4. FETCH         → Firecrawl scrapes selected pages
  5. CHUNK+FILTER  → extract relevant snippets (not full pages)
  6. SYNTHESIZE    → Claude extracts value prop + ICP from snippets only
  7. CACHE WRITE   → save results for future runs

Token discipline enforced at step 5: the synthesis call never sees more
than TOP_K_CHUNKS * ~300 tokens of content per page.
"""

import json
import os
import anthropic
from dotenv import load_dotenv

from tools.cache import cache_get, cache_set, cache_set_pages, cache_get_pages
from tools.firecrawl_client import map_site, scrape_pages
from tools.chunker import extract_relevant_snippets, format_snippets_for_prompt
from prompts.prompts import SENDER_PLAN_PROMPT, ICP_EXTRACTION_PROMPT

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _plan_fetch_targets(url: str, discovered_pages: list[str]) -> dict:
    """
    Agent planning step: Claude decides which pages to fetch and why.
    Cheap call — just page selection, not synthesis.
    """
    prompt = SENDER_PLAN_PROMPT.format(
        url=url,
        pages=json.dumps(discovered_pages[:30], indent=2)
    )
    response = _client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "pages_to_fetch": discovered_pages[:4],
            "research_goals": ["Understand the company's product and target customers"]
        }


def _synthesize_icp(url: str, snippets: list[dict]) -> dict:
    """
    Synthesis step: extract value prop and ICP from snippets only.
    This is the expensive call — but it receives only the filtered snippets,
    not full page content.
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
    # Normalize URL
    if not url.startswith("http"):
        url = "https://" + url

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

    # ── Step 2: Map site ─────────────────────────────────────────────────────
    discovered = map_site(url, limit=20)
    print(f"[sender_agent] Discovered {len(discovered)} pages")

    # ── Step 3: Plan ─────────────────────────────────────────────────────────
    plan = _plan_fetch_targets(url, discovered)
    pages_to_fetch = plan.get("pages_to_fetch", discovered[:4])
    research_goals = plan.get("research_goals", [])
    print(f"[sender_agent] Plan: fetch {len(pages_to_fetch)} pages")

    # ── Step 4: Fetch ─────────────────────────────────────────────────────────
    # Check if we have cached pages (pages fetched but ICP not yet extracted)
    cached_pages = cache_get_pages(url) or {}
    pages_to_scrape = [p for p in pages_to_fetch if p not in cached_pages]

    scraped = scrape_pages(pages_to_scrape)
    print(f"[sender_agent] Fetched {len(scraped)} new pages")

    # Merge with cached pages
    all_pages = {**cached_pages}
    for page in scraped:
        all_pages[page["url"]] = page["markdown"]
    cache_set_pages(url, all_pages)

    # Build page objects for chunker
    page_objects = [{"url": u, "markdown": m} for u, m in all_pages.items()]

    # ── Step 5: Chunk + filter ────────────────────────────────────────────────
    # Combine research goals into a single retrieval goal
    combined_goal = (
        "Value proposition, target customer profile, "
        "case studies, customer types, and differentiators. " +
        " ".join(research_goals)
    )
    snippets = extract_relevant_snippets(page_objects, combined_goal, top_k=12)
    print(f"[sender_agent] Selected {len(snippets)} relevant snippets")

    # ── Step 6: Synthesize ────────────────────────────────────────────────────
    result = _synthesize_icp(url, snippets)

    # ── Step 7: Cache write ───────────────────────────────────────────────────
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
