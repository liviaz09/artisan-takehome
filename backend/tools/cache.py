"""
cache.py — Lead memory layer.

Design rationale:
  In production this would be Postgres. Locally, a JSON file keyed by
  domain gives us the same pattern: check memory first, hit the web only
  on a miss, then write back. This avoids redundant Firecrawl + LLM calls
  on repeated runs — critical for token cost control.

Cache schema per domain:
  {
    "artisan.co": {
      "cached_at": "2025-01-01T00:00:00",   ← only set on FINAL write
      "complete": true,                       ← signals a finished run
      "type": "sender" | "target",
      "pages": { "<url>": "<markdown content>" },
      "value_prop": "...",
      "icp": { ... },
      "fit_result": { ... },
      "emails_<persona_key>": { ... },
      "claim_map": [ ... ]
    }
  }
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "leads_cache.json")
# Cache TTL: 48 hours. Stale data means stale emails.
CACHE_TTL_HOURS = 48


def _load() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save(data: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _domain_key(url: str) -> str:
    """Normalize URL to a stable cache key."""
    url = url.lower().strip()
    for prefix in ["https://", "http://", "www."]:
        url = url.removeprefix(prefix)
    return url.rstrip("/")


def _merge(url: str, updates: dict) -> None:
    """
    Merge fields into a cache entry WITHOUT updating cached_at or complete.
    Save changes to file mid-run so a second request doesn't mistake a partial entry for a complete one.
    """
    data = _load()
    key = _domain_key(url)
    existing = data.get(key, {})
    existing.update(updates)
    data[key] = existing
    _save(data)


def cache_get(url: str, field: str | None = None) -> Any | None:
    """
    Retrieve cached data for a domain.
    Returns None if:
      - No entry exists
      - Entry is not marked complete (i.e. a previous run didn't finish)
      - Entry is older than CACHE_TTL_HOURS
    """
    data = _load()
    key = _domain_key(url)
    entry = data.get(key)

    if not entry:
        return None

    # Only return entries from fully completed runs
    if not entry.get("complete"):
        return None

    # TTL check — uses cached_at which is only set on final write
    cached_at_str = entry.get("cached_at", "2000-01-01")
    try:
        cached_at = datetime.fromisoformat(cached_at_str)
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
        if age_hours > CACHE_TTL_HOURS:
            return None
    except ValueError:
        return None

    if field:
        return entry.get(field)
    return entry


def cache_set(url: str, updates: dict) -> None:
    """
    Final write: merge fields, stamp cached_at, mark complete=True.
    Only call this when the agent run has fully finished.
    """
    data = _load()
    key = _domain_key(url)
    existing = data.get(key, {})
    existing.update(updates)
    existing["cached_at"] = datetime.now(timezone.utc).isoformat()
    existing["complete"] = True
    data[key] = existing
    _save(data)


def cache_get_pages(url: str) -> dict | None:
    """
    Retrieve raw scraped pages for a domain.
    Works even on incomplete entries — we always want to reuse
    already-fetched pages to avoid redundant Firecrawl calls.
    """
    data = _load()
    key = _domain_key(url)
    entry = data.get(key)
    if not entry:
        return None
    return entry.get("pages")


def cache_set_pages(url: str, pages: dict) -> None:
    """
    Save scraped pages as an intermediate write.
    Does NOT stamp cached_at or set complete — keeps the entry
    invisible to cache_get() until the final result is written.
    """
    _merge(url, {"pages": pages})
