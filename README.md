# Outbound Intelligence Engine
### Artisan Applied AI Take-Home

A web app that turns any public company website into a complete outbound strategy — ICP definition, account research, fit evaluation, and evidence-grounded email drafts.

---

## Setup

### 1. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Open .env and fill in your keys
```

You need two keys:
- **Anthropic** — [console.anthropic.com/keys](https://console.anthropic.com/keys)
- **OpenAI** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys) (used for embeddings only)

### 3. Start the server
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open the app
Navigate to **http://localhost:8000**

---

## How to use it

**Step 1 — ICP Generation**

Enter the sender company's website (e.g. `artisan.co`). The agent researches their public pages and produces a value proposition and structured ICP — target industries, company size bands, common triggers, likely buyers, pain points, and differentiators.

**Step 2 — Outbound Drafting**

Enter a target company's website and select a recipient persona (role + seniority). The agent researches the target, evaluates how well it fits the sender's ICP with a 0–100 score, and — if the score is 50 or above — generates two outbound emails with different angles plus an evidence panel mapping every factual claim to its source.

---

## Architecture

### Agent loop

Both agents use Anthropic's native tool use API in a ReAct loop. Claude reasons about what to do next, calls a tool, observes the result, and repeats until it has enough evidence to call `finish()`.

```
User goal
  ↓
Claude reasons → calls tool → observes result
  ↓ (repeats until confident)
Claude calls finish() → structured output returned
```

Claude drives all decisions: which pages to fetch, what to search for, when it has enough evidence. The loop ends when Claude calls `finish()`, not after a fixed number of steps.

### Tools

Each agent has three tools:

| Tool | Description |
|------|-------------|
| `scrape_page(url, goal)` | Fetches a page via Jina AI Reader, chunks and embeds the content, returns only the snippets most relevant to the goal |
| `search_web(query, goal)` | Searches via DuckDuckGo, returns relevant snippets from results |
| `finish(...)` | Structured output mechanism — Claude calls this when it has enough evidence. Different schema per agent. |

### Token discipline

The `goal` parameter on `scrape_page` and `search_web` drives semantic filtering inside the tool. Pages are split into chunks via `RecursiveCharacterTextSplitter`, embedded via OpenAI `text-embedding-3-small`, and ranked by cosine similarity against the goal vector. Claude only ever sees the top-k relevant snippets — never full page content.

### Fit threshold

After the target agent calls `finish()`, the fit score is checked against a hard threshold of 50. Below 50, the response returns the fit evaluation only and skips email generation entirely. This is a business rule enforced in code, not by the agent.

---

## Design decisions

**Why Jina AI Reader instead of Firecrawl:**
Jina is free with no quota limits — prefix any URL with `r.jina.ai/` and get clean markdown back. No API key required. In production, Firecrawl or a similar service would give higher reliability on complex JS-heavy sites.

**Why DuckDuckGo instead of a search API:**
Every search API worth using has either a cost or a tight free tier. DuckDuckGo's HTML endpoint requires no authentication and has no limits. In production this would be replaced with a proper search API for cleaner structured results.

**Why no LangChain agent framework:**
The agent loop is implemented directly using Anthropic's tool use API. This keeps every step explicit, makes the code readable without framework knowledge, and gives full control over error handling and token usage. LangChain would abstract the tool loop behind framework conventions that are harder to explain and debug.

**Why no caching:**
Caching is an infrastructure concern, not an intelligence concern. For a local demo it adds complexity without meaningfully improving what's being evaluated. In production: Postgres keyed by domain with a 24-hour TTL, invalidated on re-run.
