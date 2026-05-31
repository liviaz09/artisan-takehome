"""
chunker.py — Snippet extraction and relevance filtering.

Design rationale:
  This is the core of our token optimization strategy. The requirement says
  "answers must be based on retrieved snippets, not full-context stuffing."

  The pattern:
    1. Split page markdown into ~300 token chunks (≈ 1,200 chars)
    2. Score each chunk for relevance to a goal using a CHEAP, fast Claude call
       (haiku-class model, low max_tokens)
    3. Return only top-k chunks to the expensive synthesis call

  This means the synthesis prompt sees ~3,000 tokens of signal, not
  30,000 tokens of full page content. Same RAG principle as a vector DB,
  right-sized for a local app without the embedding infrastructure.

  In production: replace step 2 with cosine similarity over embeddings
  in a vector store. The interface (goal → relevant_chunks) stays identical.
"""

import re
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CHUNK_SIZE_CHARS = 1200   # ~300 tokens at avg 4 chars/token
CHUNK_OVERLAP_CHARS = 150 # overlap so we don't split mid-sentence
MAX_CHUNKS_TO_SCORE = 40  # don't score more than this per page (cost guard)
TOP_K_CHUNKS = 5          # chunks returned per page to synthesis


def chunk_markdown(markdown: str, url: str) -> list[dict]:
    """
    Split markdown into overlapping chunks.
    Returns list of { text, url, chunk_index } dicts.
    """
    # Clean up excessive whitespace while preserving structure
    text = re.sub(r'\n{3,}', '\n\n', markdown.strip())
    text = re.sub(r' {2,}', ' ', text)

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + CHUNK_SIZE_CHARS

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break first
            para_break = text.rfind('\n\n', start, end)
            sent_break = max(text.rfind('. ', start, end),
                             text.rfind('! ', start, end),
                             text.rfind('? ', start, end))

            if para_break > start + CHUNK_SIZE_CHARS // 2:
                end = para_break
            elif sent_break > start + CHUNK_SIZE_CHARS // 2:
                end = sent_break + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "url": url,
                "chunk_index": idx,
            })
            idx += 1

        start = end - CHUNK_OVERLAP_CHARS
        if start >= len(text):
            break

    return chunks


def score_chunks_for_relevance(chunks: list[dict], goal: str) -> list[dict]:
    """
    Use a fast Claude call to score chunks for relevance to a goal.
    Returns chunks sorted by relevance score, descending.

    This is a single LLM call that scores all chunks at once —
    much cheaper than one call per chunk.
    """
    if not chunks:
        return []

    # Cap to MAX_CHUNKS_TO_SCORE for cost control
    chunks_to_score = chunks[:MAX_CHUNKS_TO_SCORE]

    numbered = "\n\n".join(
        f"[{i}] {c['text'][:600]}" for i, c in enumerate(chunks_to_score)
    )

    prompt = f"""You are scoring text chunks for relevance to a research goal.

Goal: {goal}

Chunks:
{numbered}

Return ONLY a JSON array of objects, one per chunk, in this exact format:
[{{"index": 0, "score": 8}}, {{"index": 1, "score": 3}}, ...]

Score 0-10 where 10 = highly relevant, 0 = irrelevant. No explanation."""

    try:
        response = _client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()

        # Parse scores
        import json
        # Handle potential markdown code fences
        raw = re.sub(r'```json?\s*|\s*```', '', raw).strip()
        scores = json.loads(raw)

        score_map = {item["index"]: item["score"] for item in scores}
        for i, chunk in enumerate(chunks_to_score):
            chunk["relevance_score"] = score_map.get(i, 0)

        return sorted(chunks_to_score, key=lambda c: c.get("relevance_score", 0), reverse=True)

    except Exception as e:
        print(f"[chunker] scoring error: {e}")
        # Fallback: return chunks in original order with neutral score
        for chunk in chunks_to_score:
            chunk["relevance_score"] = 5
        return chunks_to_score


def extract_relevant_snippets(pages: list[dict], goal: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    """
    Main entry point. Given a list of scraped pages and a research goal,
    returns the top-k most relevant snippets across all pages.

    Each snippet: { text, url, chunk_index, relevance_score }
    These snippets — and only these — go into the synthesis LLM call.
    """
    all_chunks = []
    for page in pages:
        chunks = chunk_markdown(page["markdown"], page["url"])
        all_chunks.extend(chunks)

    if not all_chunks:
        return []

    scored = score_chunks_for_relevance(all_chunks, goal)
    return scored[:top_k]


def format_snippets_for_prompt(snippets: list[dict]) -> str:
    """
    Format snippets for injection into a synthesis prompt.
    Each snippet tagged with its source URL for claim mapping.
    """
    lines = []
    for i, s in enumerate(snippets):
        lines.append(f"[SOURCE {i+1}: {s['url']}]\n{s['text']}")
    return "\n\n---\n\n".join(lines)
