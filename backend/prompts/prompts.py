"""
prompts.py — All LLM prompts in one place.

Design rationale:
  Centralizing prompts makes them easy to iterate on, version, and discuss
  in the review. Each prompt has a clear role, tone, constraints, and
  output format — no few-shot examples (which would kill email variability).

  Prompt structure per call:
    Role        → who Claude is in this context
    Goal        → specific task
    Constraints → what NOT to do (as important as what to do)
    Format      → exact output shape (JSON schema described inline)
"""

# ---------------------------------------------------------------------------
# PLANNING PROMPTS
# ---------------------------------------------------------------------------

SENDER_PLAN_PROMPT = """You are a B2B market intelligence agent planning a research task.

Given a company's website, identify the most valuable pages to scrape to understand:
1. What the company sells and to whom
2. Their value proposition and differentiation  
3. Evidence of their ideal customer (case studies, testimonials, customer logos)
4. Pricing signals (which segments they target)

Website: {url}
Discovered pages: {pages}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "pages_to_fetch": ["url1", "url2"],
  "research_goals": [
    "Understand core product and value proposition",
    "Identify target customer segments",
    "Find ICP signals from case studies and testimonials"
  ]
}}

Select at most 6 pages. Prioritize: homepage, /about, /customers or /case-studies, /pricing, /product."""


TARGET_PLAN_PROMPT = """You are a B2B sales intelligence agent planning account research.

Given a target company's website and a recipient persona, identify:
1. What the company does and their industry
2. Company size signals (team pages, job postings, funding)
3. Tech stack or tooling signals
4. Recent triggers: funding, launches, hiring surges, leadership changes
5. Pain points relevant to the sender's ICP

Website: {url}
Discovered pages: {pages}
Recipient persona: {role} ({seniority})
Sender ICP context: {icp_summary}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "pages_to_fetch": ["url1", "url2"],
  "web_searches": [
    "{{company name}} funding 2024",
    "{{company name}} recent news"
  ],
  "signals_to_find": [
    "Company size and growth stage",
    "Industry and sub-vertical",
    "Recent business triggers",
    "Tech stack or tooling"
  ]
}}

Select at most 5 pages and at most 3 web searches."""


# ---------------------------------------------------------------------------
# MODE 1: ICP + VALUE PROPOSITION
# ---------------------------------------------------------------------------

ICP_EXTRACTION_PROMPT = """You are a senior GTM strategist analyzing a B2B company to define their ideal customer profile.

Your analysis must be grounded ONLY in the retrieved evidence below. Do not invent claims.
Every insight must map to a specific source snippet.

Company website: {url}

Retrieved evidence:
{snippets}

Extract the following and return ONLY valid JSON (no markdown, no explanation):
{{
  "value_proposition": "One clear sentence: what the company does, for whom, and the primary outcome they deliver.",
  "icp": {{
    "target_industries": ["industry1", "industry2"],
    "size_bands": ["e.g. 50-500 employees", "Series A to Series C"],
    "common_triggers": ["e.g. scaling sales team", "just raised funding", "replacing manual process"],
    "likely_buyers": ["e.g. VP of Sales", "Head of Revenue Operations"],
    "pain_points": ["pain1", "pain2"],
    "differentiators": ["what sets this company apart from alternatives"]
  }},
  "confidence": "high | medium | low",
  "evidence_gaps": ["anything you couldn't determine from the evidence"]
}}

Be specific and concrete. Generic answers like 'B2B companies' are not acceptable.
If evidence is thin, say so in evidence_gaps and lower confidence."""


# ---------------------------------------------------------------------------
# MODE 2: TARGET FIT EVALUATION
# ---------------------------------------------------------------------------

FIT_EVALUATION_PROMPT = """You are a B2B sales intelligence agent evaluating how well a target company fits a sender's ICP.

Sender ICP:
{icp}

Target company: {target_url}
Recipient: {role} ({seniority})

Retrieved evidence about target:
{snippets}

Evaluate fit and return ONLY valid JSON (no markdown, no explanation):
{{
  "fit_score": 0-100,
  "fit_label": "Strong Fit | Good Fit | Partial Fit | Poor Fit",
  "fit_summary": "2-3 sentence explanation of why this target fits or doesn't",
  "matched_signals": [
    {{"signal": "description of match", "evidence": "quote or paraphrase", "source_url": "url"}}
  ],
  "gap_signals": ["ICP criteria this target doesn't clearly match"],
  "company_profile": {{
    "name": "company name",
    "industry": "specific industry",
    "estimated_size": "e.g. 200-500 employees",
    "stage": "e.g. Series B startup",
    "recent_triggers": ["trigger1", "trigger2"],
    "tech_signals": ["tool or tech stack signals found"]
  }}
}}"""


# ---------------------------------------------------------------------------
# MODE 2: EMAIL GENERATION
# ---------------------------------------------------------------------------

EMAIL_GENERATION_PROMPT = """You are an expert B2B outbound copywriter. Write two cold emails that are meaningfully different in angle and approach.

CONTEXT:
Sender company: {sender_url}
Sender value proposition: {value_prop}
Target company: {target_url}
Target company profile: {company_profile}
Recipient: {role} ({seniority})
ICP fit summary: {fit_summary}
Matched signals: {matched_signals}

RETRIEVED EVIDENCE (all factual claims must trace to this):
{snippets}

INSTRUCTIONS:
Write Email A (pain-led) and Email B (trigger-led). 

Email A — Pain-led:
- Open by naming a specific pain or challenge the recipient's company likely faces
- Connect that pain to the sender's solution
- Use evidence from the snippets to make it feel researched, not templated

Email B — Trigger-led:
- Open by referencing a specific recent trigger (funding, launch, hire, growth signal)
- Frame the sender's solution as timely given that trigger
- Use evidence from the snippets to make it specific

RULES (non-negotiable):
- Subject lines: under 8 words, no clickbait, no exclamation marks
- Body: 4-6 sentences max. Brevity is respect.
- No generic openers ("I hope this finds you well", "I came across your company")
- No feature lists. One insight, one connection, one ask.
- CTA: specific, low-friction (e.g. "Worth a 15-min call this week?")
- Every factual claim about the target must come from the evidence above
- Sign off with a human name (use "Alex" as placeholder)

Return ONLY valid JSON (no markdown, no explanation):
{{
  "email_a": {{
    "angle": "pain-led",
    "subject": "...",
    "body": "...",
    "claims": [
      {{"claim": "factual statement used in email", "source_url": "url", "snippet": "supporting text from evidence"}}
    ]
  }},
  "email_b": {{
    "angle": "trigger-led", 
    "subject": "...",
    "body": "...",
    "claims": [
      {{"claim": "factual statement used in email", "source_url": "url", "snippet": "supporting text from evidence"}}
    ]
  }}
}}"""
