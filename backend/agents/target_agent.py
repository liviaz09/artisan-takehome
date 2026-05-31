"""
target_agent.py — Mode 2: target evaluation and outbound email drafting.

Agent loop:
  1. CHECK CACHE   → return if we've already profiled this target+persona
  2. MAP + SELECT  → discover target's pages, filter deterministically
  3. FETCH         → Firecrawl scrapes pages + hardcoded web searches
  4. CHUNK+EMBED   → RecursiveCharacterTextSplitter + text-embedding-3-small
  5. EVALUATE FIT  → Claude scores target against sender's ICP
  6. DRAFT EMAILS  → Claude writes email A (pain-led) + B (trigger-led)
  7. BUILD CLAIM MAP → every factual claim → source URL + snippet
  8. CACHE WRITE   → save for future runs

What we removed vs v1:
  - The Claude planning call (_plan_target_research) is gone.
    We hardcode the web search queries because we already know what signals
    matter: funding, hiring, news. Claude shouldn't decide this.

  - Page selection is deterministic (select_pages), not LLM-driven.

  - Research goal is a hardcoded string — we know what we're looking for.
"""

import json
import os
import anthropic
from dotenv import load_dotenv

from tools.cache import cache_get, cache_set, cache_set_pages, cache_get_pages
from tools.firecrawl_client import map_site, scrape_pages, search_web, select_pages
from tools.chunker import extract_relevant_snippets, format_snippets_for_prompt
from prompts.prompts import FIT_EVALUATION_PROMPT, EMAIL_GENERATION_PROMPT

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Hardcoded research goal for target signal extraction.
# These are the signals that always matter for ICP fit — no LLM needed to decide.
TARGET_RESEARCH_GOAL = (
    "company size, employee count, industry vertical, business model B2B or B2C, "
    "funding stage, recent funding rounds, revenue or growth signals, "
    "sales team existence, outbound sales motion, tech stack, "
    "pain points, hiring signals, recent news or press releases"
)


def _build_web_searches(company_name: str) -> list[str]:
    """
    Hardcoded web search queries for external signals.
    We know exactly what external signals matter — funding, hiring, news.
    No LLM needed to generate these queries.
    """
    return [
        f"{company_name} funding round 2024 2025",
        f"{company_name} sales team hiring",
        f"{company_name} recent news announcement",
    ]


def _extract_company_name(url: str) -> str:
    """Best-effort company name from URL for search queries."""
    domain = url.split("://")[-1].split("/")[0]
    domain = domain.removeprefix("www.")
    return domain.split(".")[0].capitalize()


def _get_sender_icp(sender_url: str) -> dict:
    """Retrieve sender's ICP from cache (computed in Mode 1)."""
    if not sender_url.startswith("http"):
        sender_url = "https://" + sender_url
    cached = cache_get(sender_url)
    if cached:
        return {
            "value_prop": cached.get("value_prop", ""),
            "icp": cached.get("icp", {}),
        }
    return {}


def _evaluate_fit(
    target_url: str,
    role: str,
    seniority: str,
    icp: dict,
    snippets: list[dict],
) -> dict:
    """Claude evaluates how well the target fits the sender's ICP."""
    formatted = format_snippets_for_prompt(snippets)
    prompt = FIT_EVALUATION_PROMPT.format(
        icp=json.dumps(icp, indent=2),
        target_url=target_url,
        role=role,
        seniority=seniority,
        snippets=formatted,
    )
    response = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"[target_agent] fit eval parse error: {e}")
        return {"fit_score": 50, "fit_label": "Unknown", "fit_summary": "Could not evaluate."}


def _draft_emails(
    sender_url: str,
    value_prop: str,
    target_url: str,
    role: str,
    seniority: str,
    fit_result: dict,
    snippets: list[dict],
) -> dict:
    """Claude drafts two emails with different angles from retrieved evidence."""
    formatted = format_snippets_for_prompt(snippets)
    prompt = EMAIL_GENERATION_PROMPT.format(
        sender_url=sender_url,
        value_prop=value_prop,
        target_url=target_url,
        company_profile=json.dumps(fit_result.get("company_profile", {}), indent=2),
        role=role,
        seniority=seniority,
        fit_summary=fit_result.get("fit_summary", ""),
        matched_signals=json.dumps(fit_result.get("matched_signals", []), indent=2),
        snippets=formatted,
    )
    response = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"[target_agent] email parse error: {e}\nRaw: {raw[:300]}")
        return {"email_a": {}, "email_b": {}}


async def analyze_target(
    sender_url: str,
    target_url: str,
    role: str,
    seniority: str,
) -> dict:
    """
    Main entry point for Mode 2.
    Returns: { fit_result, email_a, email_b, claim_map, snippets_used, cached }
    """
    if not target_url.startswith("http"):
        target_url = "https://" + target_url
    if not sender_url.startswith("http"):
        sender_url = "https://" + sender_url
    base = target_url.rstrip("/")

    cache_persona_key = f"{role}_{seniority}".lower().replace(" ", "_").replace("/", "_")

    # ── Step 1: Cache check ──────────────────────────────────────────────────
    cached = cache_get(target_url)
    if cached and cached.get(f"emails_{cache_persona_key}"):
        print(f"[target_agent] Cache hit for {target_url} + {cache_persona_key}")
        email_data = cached[f"emails_{cache_persona_key}"]
        return {
            "fit_result": cached.get("fit_result", {}),
            "email_a": email_data.get("email_a", {}),
            "email_b": email_data.get("email_b", {}),
            "claim_map": email_data.get("claim_map", []),
            "snippets_used": cached.get("snippets_used", []),
            "pages_fetched": list(cached.get("pages", {}).keys()),
            "cached": True,
        }

    print(f"[target_agent] Cache miss — starting agent loop for {target_url}")

    # ── Tool B: Get sender ICP from cache ─────────────────────────────────────
    sender_data = _get_sender_icp(sender_url)
    icp = sender_data.get("icp", {})
    value_prop = sender_data.get("value_prop", "")

    # ── Step 2: Map + deterministic page selection ────────────────────────────
    discovered = map_site(target_url, limit=30)
    pages_to_fetch = select_pages(discovered, base, mode="target", limit=5)
    print(f"[target_agent] Selected {len(pages_to_fetch)} pages: {pages_to_fetch}")

    # ── Step 3: Fetch pages + hardcoded web searches ──────────────────────────
    cached_pages = cache_get_pages(target_url) or {}
    pages_to_scrape = [p for p in pages_to_fetch if p not in cached_pages]
    scraped = scrape_pages(pages_to_scrape)

    all_pages = {**cached_pages}
    for page in scraped:
        all_pages[page["url"]] = page["markdown"]
    cache_set_pages(target_url, all_pages)

    # Hardcoded web searches — we know what external signals matter
    company_name = _extract_company_name(target_url)
    search_queries = _build_web_searches(company_name)
    search_results = []
    for query in search_queries:
        results = search_web(query, limit=3)
        search_results.extend(results)
    print(f"[target_agent] Got {len(search_results)} web search results for: {company_name}")

    # Combine site pages + search results
    page_objects = [{"url": u, "markdown": m} for u, m in all_pages.items()]
    page_objects.extend(search_results)

    # ── Step 4: Chunk + embed + similarity ranking ────────────────────────────
    snippets = extract_relevant_snippets(page_objects, TARGET_RESEARCH_GOAL, top_k=15)
    print(f"[target_agent] Selected {len(snippets)} relevant snippets via embeddings")

    # ── Step 5: Evaluate fit (one Claude call) ────────────────────────────────
    fit_result = _evaluate_fit(target_url, role, seniority, icp, snippets)
    print(f"[target_agent] Fit score: {fit_result.get('fit_score')} — {fit_result.get('fit_label')}")

    # ── Step 6: Draft emails (one Claude call) ────────────────────────────────
    email_data = _draft_emails(
        sender_url, value_prop, target_url, role, seniority, fit_result, snippets
    )

    # ── Step 7: Build claim map ───────────────────────────────────────────────
    claim_map = []
    for email_key in ["email_a", "email_b"]:
        email = email_data.get(email_key, {})
        for claim in email.get("claims", []):
            claim_map.append({
                "email": email_key,
                "angle": email.get("angle", email_key),
                "claim": claim.get("claim", ""),
                "source_url": claim.get("source_url", ""),
                "snippet": claim.get("snippet", ""),
            })

    # ── Step 8: Cache write ───────────────────────────────────────────────────
    cache_set(target_url, {
        "type": "target",
        "fit_result": fit_result,
        "snippets_used": snippets,
        f"emails_{cache_persona_key}": {
            "email_a": email_data.get("email_a", {}),
            "email_b": email_data.get("email_b", {}),
            "claim_map": claim_map,
        },
    })

    return {
        "fit_result": fit_result,
        "email_a": email_data.get("email_a", {}),
        "email_b": email_data.get("email_b", {}),
        "claim_map": claim_map,
        "snippets_used": snippets,
        "pages_fetched": list(all_pages.keys()),
        "cached": False,
    }
