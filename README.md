# Outbound Intelligence Engine
### Artisan Applied AI Take-Home

A web app that turns public company information into outbound strategy — built with an agentic retrieval pipeline, token-efficient snippet extraction, and evidence-grounded email generation.

---

## Quick Start

### 1. Clone & install

```bash
cd artisan-takehome
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your keys:
# ANTHROPIC_API_KEY=sk-ant-...
# FIRECRAWL_API_KEY=fc-...
```

Get a Firecrawl API key (free, 1000 credits/month): https://firecrawl.dev

### 3. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open the app

Navigate to: **http://localhost:8000**

---

## Architecture

### Agent Loop

```
INPUT: Company URL
       ↓
[1] CACHE CHECK     → JSON file keyed by domain (24hr TTL)
    Hit?  → return immediately (0 tokens used)
    Miss? → continue
       ↓
[2] MAP SITE        → Firecrawl /map discovers all URLs on the site
       ↓
[3] PLAN            → Claude Haiku: "which pages matter for this goal?"
                      Returns structured page list + research goals
                      (cheap call, ~150 tokens output)
       ↓
[4] FETCH           → Firecrawl /scrape: JS-rendered clean Markdown
                      Tool A (live web) + Tool B (cache for repeat pages)
       ↓
[5] CHUNK + FILTER  → Pages split into ~300-token overlapping chunks
                      Claude Haiku scores chunks for relevance (0-10)
                      Top-k chunks selected (synthesis never sees full pages)
       ↓
[6] SYNTHESIZE      → Claude Sonnet over snippets only:
                      Mode 1: value prop + ICP JSON
                      Mode 2: fit score + email A + email B + claim map
       ↓
[7] CACHE WRITE     → Result saved to leads_cache.json
```

### Two-Tool Pattern

| Tool | Purpose | When used |
|------|---------|-----------|
| **Firecrawl** (live web) | JS-rendered page content, web-wide search | Cache miss, new pages |
| **JSON cache** | Memory of previously analyzed companies | Cache hit, sender ICP lookup |

### Token Optimization

The core constraint: *"answers must be based on retrieved snippets, not full-context stuffing."*

How we enforce it:
- Pages average ~8,000 tokens of raw content
- After chunking + relevance filtering, synthesis sees ~2,000-3,000 tokens
- Planning calls use Claude Haiku (cheaper, faster) — only synthesis uses Sonnet
- Cache eliminates all LLM calls on repeat runs

### Why Firecrawl

Most company websites are JS-rendered SPAs. A naive HTTP fetch returns an empty `<div id="root">`. Firecrawl handles JS rendering and returns clean Markdown — no BeautifulSoup, no Playwright, no HTML parsing. This directly reduces token noise and setup friction.

### Why No Vector DB

For <10 pages per run, cosine similarity over embeddings adds latency and infrastructure complexity with no measurable quality benefit over scored chunk selection. The interface is identical: `goal → relevant_chunks`. In production, this step becomes embeddings + vector store with no changes to the agent loop.

### Why Manual Agent Loop (No LangChain)

Control and transparency. Every step is explicit, every token cost is visible, every prompt is accessible. This is what you'd build in production when you own reliability and cost.

---

## File Structure

```
artisan-takehome/
├── backend/
│   ├── main.py                  # FastAPI routes (thin layer)
│   ├── leads_cache.json         # Created on first run
│   ├── agents/
│   │   ├── sender_agent.py      # Mode 1: ICP + value prop
│   │   └── target_agent.py      # Mode 2: fit eval + email draft
│   ├── tools/
│   │   ├── firecrawl_client.py  # Firecrawl wrapper (Tool A)
│   │   ├── chunker.py           # Chunk + relevance filter
│   │   └── cache.py             # JSON cache (Tool B)
│   └── prompts/
│       └── prompts.py           # All LLM prompts, centralized
└── frontend/
    └── index.html               # Single-file React app
```

---

## Design Decisions

### Email angles
- **Pain-led (Email A)**: Opens by naming a specific inferred pain point. Works when signals point to a known struggle (job postings mentioning a problem, competitor mentions, growth friction).
- **Trigger-led (Email B)**: Opens with a recent signal (funding, product launch, hiring surge). Works when there's a clear moment of change — timing makes the email feel less cold.

### No few-shot prompting for emails
Few-shot examples anchor the model to a style, killing the variability that makes A/B emails worth comparing. Rich instructions (role, constraints, angle) without examples preserve creative range.

### Claim map
Every factual claim in both emails is tagged with a source URL and supporting snippet. This makes the agent's reasoning auditable and prevents hallucinated claims from reaching the output.

---

## In Production

| Local (this demo) | Production |
|---|---|
| JSON file cache | Postgres + Redis |
| Claude Haiku for planning/scoring | Same or cheaper model |
| Claude Sonnet for synthesis | Same |
| Sequential page fetching | Parallel async fetches |
| Firecrawl scrape | Firecrawl (same API) |
| Chunk relevance scoring via LLM | Embeddings + vector store |
