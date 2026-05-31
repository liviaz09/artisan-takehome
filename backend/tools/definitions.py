"""
tools/definitions.py — Anthropic tool schemas.

These are the JSON schemas passed to Claude in every API call.
They define the contract between Claude and the tool implementations.

Separation rationale:
  Definitions are what Claude sees — they change when we want Claude
  to understand a tool differently (description, parameters).
  Implementations are what Python executes — they change when we want
  different behavior (different API, different filtering logic).
  They should be independently readable and modifiable.

The 'goal' parameter on scrape_page and search_web is critical —
it tells Claude that what gets returned depends on what it's looking for.
Claude must pass a specific research goal, not a generic string,
because the goal drives the embedding relevance filter inside the tool.
"""

# ── Shared tools (used by both agents) ──────────────────────────────────────

SCRAPE_PAGE_TOOL = {
    "name": "scrape_page",
    "description": (
        "Fetches a webpage and returns the most relevant snippets for your current research goal. "
        "The tool internally filters the page content using semantic similarity — "
        "you will only receive content relevant to the goal you specify. "
        "Use this to research a company's own website: homepage, about, pricing, customers, blog posts. "
        "Always pass a specific goal describing what you are trying to learn from this page."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch, including https://"
            },
            "goal": {
                "type": "string",
                "description": (
                    "What you are trying to learn from this page. "
                    "Be specific — this drives what content gets returned. "
                    "Example: 'target industries and company size of customers' "
                    "or 'recent funding rounds and growth signals'"
                )
            }
        },
        "required": ["url", "goal"]
    }
}

SEARCH_WEB_TOOL = {
    "name": "search_web",
    "description": (
        "Searches the web and returns relevant snippets from the top results. "
        "Use this for information not on the company's own website: "
        "funding announcements, press coverage, hiring signals, news, competitor mentions. "
        "The tool filters results by semantic similarity to your goal. "
        "Always pass a specific goal describing what signal you are looking for."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific — include company name and topic."
            },
            "goal": {
                "type": "string",
                "description": (
                    "What signal you are looking for in the results. "
                    "Example: 'Series B funding announcement and amount raised' "
                    "or 'recent executive hires in sales leadership'"
                )
            }
        },
        "required": ["query", "goal"]
    }
}


# ── Sender agent finish tool ─────────────────────────────────────────────────

SENDER_FINISH_TOOL = {
    "name": "finish",
    "description": (
        "Call this when you have gathered sufficient evidence to define "
        "a clear value proposition and ICP. "
        "Do not call this until you have researched at least the homepage "
        "and one additional page with customer or product evidence. "
        "All fields must be grounded in evidence you have retrieved — "
        "do not invent or assume information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "value_proposition": {
                "type": "string",
                "description": (
                    "One clear sentence: what the company does, for whom, "
                    "and the primary outcome they deliver. "
                    "Must be specific — not generic like 'helps businesses grow'."
                )
            },
            "icp": {
                "type": "object",
                "description": "Structured ideal customer profile",
                "properties": {
                    "target_industries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific industries the company targets"
                    },
                    "size_bands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Company size ranges e.g. '50-500 employees', 'Series A to C'"
                    },
                    "common_triggers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Events that make a company likely to buy e.g. 'scaling sales team', 'just raised Series A'"
                    },
                    "likely_buyers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Job titles of the people who buy this product"
                    },
                    "pain_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Core problems this product solves"
                    },
                    "differentiators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What makes this company different from alternatives"
                    }
                },
                "required": [
                    "target_industries", "size_bands", "common_triggers",
                    "likely_buyers", "pain_points", "differentiators"
                ]
            }
        },
        "required": ["value_proposition", "icp"]
    }
}


# ── Target agent finish tool ─────────────────────────────────────────────────

TARGET_FINISH_TOOL = {
    "name": "finish",
    "description": (
        "Call this when you have gathered sufficient evidence to evaluate "
        "how well the target company fits the sender's ICP. "
        "Do not call this until you have researched the target's homepage, "
        "at least one additional page, and run at least one web search for external signals. "
        "All claims must be grounded in evidence you retrieved — no assumptions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fit_score": {
                "type": "integer",
                "description": "ICP fit score from 0-100. 0=no fit, 100=perfect fit.",
                "minimum": 0,
                "maximum": 100
            },
            "fit_label": {
                "type": "string",
                "enum": ["Strong Fit", "Good Fit", "Partial Fit", "Poor Fit"],
                "description": "Qualitative fit label corresponding to the score"
            },
            "fit_summary": {
                "type": "string",
                "description": "2-3 sentences explaining why this target fits or doesn't fit the ICP"
            },
            "matched_signals": {
                "type": "array",
                "description": "ICP criteria this target clearly matches, with evidence",
                "items": {
                    "type": "object",
                    "properties": {
                        "signal": {"type": "string"},
                        "evidence": {"type": "string"},
                        "source_url": {"type": "string"}
                    },
                    "required": ["signal", "evidence", "source_url"]
                }
            },
            "gap_signals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ICP criteria this target does not clearly match"
            },
            "company_profile": {
                "type": "object",
                "properties": {
                    "name":             {"type": "string"},
                    "industry":         {"type": "string"},
                    "estimated_size":   {"type": "string"},
                    "stage":            {"type": "string"},
                    "recent_triggers":  {"type": "array", "items": {"type": "string"}},
                    "tech_signals":     {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "industry", "estimated_size", "stage"]
            },
            "evidence_snippets": {
                "type": "array",
                "description": "All snippets used as evidence, for claim mapping in emails",
                "items": {
                    "type": "object",
                    "properties": {
                        "text":       {"type": "string"},
                        "source_url": {"type": "string"}
                    },
                    "required": ["text", "source_url"]
                }
            }
        },
        "required": [
            "fit_score", "fit_label", "fit_summary",
            "matched_signals", "gap_signals", "company_profile", "evidence_snippets"
        ]
    }
}


# ── Tool lists passed to each agent ─────────────────────────────────────────

SENDER_TOOLS = [SCRAPE_PAGE_TOOL, SEARCH_WEB_TOOL, SENDER_FINISH_TOOL]
TARGET_TOOLS = [SCRAPE_PAGE_TOOL, SEARCH_WEB_TOOL, TARGET_FINISH_TOOL]
