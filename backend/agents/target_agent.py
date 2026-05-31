"""
target_agent.py — Mode 2: target evaluation and outbound email drafting.

Agent loop:
  1. CHECK CACHE   → return if we've already profiled this target+persona
  2. MAP SITE      → discover target's pages
  3. PLAN          → Claude decides pages + web searches based on ICP context
  4. FETCH         → Firecrawl scrapes pages + searches for external signals
  5. CHUNK+FILTER  → extract relevant snippets (fit signals, triggers, pain)
  6. EVALUATE FIT  → Claude scores target against sender's ICP
  7. DRAFT EMAILS  → Claude writes email A (pain-led) + B (trigger-led)
  8. BUILD CLAIM MAP → every factual claim → source URL + snippet
  9. CACHE WRITE   → save for future runs

The two-tool pattern:
  Tool A: Firecrawl (live web) — for pages and web-wide signals
  Tool B: JSON cache — for sender's ICP (already computed in Mode 1)
"""

import json
import os
import anthropic
from dotenv import load_dotenv

from tools.cache import cache_get, cache_set, cache_set_pages, cache_get_pages
from tools.firecrawl_client import map_site, scrape_pages, search_web
from tools.chunker import extract_relevant_snippets, format_snippets_for_prompt
from prompts.prompts import (
    TARGET_PLAN_PROMPT,
    FIT_EVALUATION_PROMPT,
    EMAIL_GENERATION_PROMPT,
)

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _get_sender_icp(sender_url: str) -> dict:
    """
    Tool B: retrieve the sender's ICP from cache (computed in Mode 1).
    If not cached, returns an empty dict — the agent degrades gracefully.
    """
    if not sender_url.startswith("http"):
        sender_url = "https://" + sender_url
    cached = cache_get(sender_url)
    if cached:
        return {
            "value_prop": cached.get("value_prop", ""),
            "icp": cached.get("icp", {}),
        }
    return {}


def _plan_target_research(
    target_url: str,
    discovered: list[str],
    role: str,
    seniority: str,
    icp_summary: str,
) -> dict:
    """Planning step: Claude selects pages to fetch + web searches to run."""
    prompt = TARGET_PLAN_PROMPT.format(
        url=target_url,
        pages=json.dumps(discovered[:30], indent=2),
        role=role,
        seniority=seniority,
        icp_summary=icp_summary,
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
            "pages_to_fetch": discovered[:4],
            "web_searches": [],
            "signals_to_find": ["Company size", "Industry", "Recent activity"],
        }


def _evaluate_fit(
    target_url: str,
    role: str,
    seniority: str,
    icp: dict,
    snippets: list[dict],
) -> dict:
    """Evaluate how well the target fits the sender's ICP."""
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
    """Draft two emails with different angles from the retrieved evidence."""
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

    # Cache key includes persona so different roles get different emails
    cache_persona_key = f"{role}_{seniority}".lower().replace(" ", "_")

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
    icp_summary = json.dumps(icp, indent=2) if icp else "ICP not yet computed for sender."

    # ── Step 2: Map target site ───────────────────────────────────────────────
    discovered = map_site(target_url, limit=20)
    print(f"[target_agent] Discovered {len(discovered)} pages on target")

    # ── Step 3: Plan ──────────────────────────────────────────────────────────
    plan = _plan_target_research(target_url, discovered, role, seniority, icp_summary)
    pages_to_fetch = plan.get("pages_to_fetch", discovered[:4])
    web_searches = plan.get("web_searches", [])
    print(f"[target_agent] Plan: {len(pages_to_fetch)} pages + {len(web_searches)} searches")

    # ── Step 4: Fetch (Tool A: Firecrawl) ────────────────────────────────────
    cached_pages = cache_get_pages(target_url) or {}
    pages_to_scrape = [p for p in pages_to_fetch if p not in cached_pages]
    scraped = scrape_pages(pages_to_scrape)

    all_pages = {**cached_pages}
    for page in scraped:
        all_pages[page["url"]] = page["markdown"]
    cache_set_pages(target_url, all_pages)

    # Fetch web search results for external signals
    search_results = []
    for query in web_searches[:3]:  # cap at 3 searches (cost control)
        results = search_web(query, limit=3)
        search_results.extend(results)
    print(f"[target_agent] Got {len(search_results)} web search results")

    # Combine site pages + search results
    page_objects = [{"url": u, "markdown": m} for u, m in all_pages.items()]
    page_objects.extend(search_results)

    # ── Step 5: Chunk + filter ────────────────────────────────────────────────
    fit_goal = (
        f"Company size, industry, growth stage, recent funding or news, "
        f"tech stack, pain points relevant to: {icp_summary[:300]}"
    )
    snippets = extract_relevant_snippets(page_objects, fit_goal, top_k=15)
    print(f"[target_agent] Selected {len(snippets)} relevant snippets")

    # ── Step 6: Evaluate fit ──────────────────────────────────────────────────
    fit_result = _evaluate_fit(target_url, role, seniority, icp, snippets)
    print(f"[target_agent] Fit score: {fit_result.get('fit_score')} — {fit_result.get('fit_label')}")

    # ── Step 7: Draft emails ──────────────────────────────────────────────────
    email_data = _draft_emails(
        sender_url, value_prop, target_url, role, seniority, fit_result, snippets
    )

    # ── Step 8: Build claim map ───────────────────────────────────────────────
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

    # ── Step 9: Cache write ───────────────────────────────────────────────────
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
