"""
chunker.py — Snippet extraction using embeddings and cosine similarity.

Pipeline (corrected):
  1. Split markdown into chunks via RecursiveCharacterTextSplitter (LangChain)
     — battle-tested library, handles markdown structure aware splitting
  2. Embed all chunks via OpenAI text-embedding-3-small
     — lightweight embedding model, not an LLM, extremely cheap
     — ~$0.00002 per 1K tokens vs ~$0.003 for Haiku
  3. Embed the research goal (same model)
  4. Cosine similarity between goal vector and all chunk vectors
  5. Return top-k chunks above relevance threshold

Why this is correct vs the previous approach:
  - Before: we sent chunks to Claude Haiku to score relevance — wasteful,
    slow, and using an LLM for a task that doesn't need language reasoning
  - Now: embedding models produce dense vectors purpose-built for semantic
    similarity. One API call embeds everything. No LLM tokens consumed here.
  - Claude is reserved for analysis only (ICP extraction, fit eval, emails)

Token cost comparison per run:
  Before: ~2,000 tokens (Haiku) just for chunk scoring
  After:  ~0 LLM tokens — embedding model handles it entirely
"""

import os
import numpy as np
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 400        # tokens (RecursiveCharacterTextSplitter uses chars internally)
CHUNK_OVERLAP = 50      # token overlap between chunks
RELEVANCE_THRESHOLD = 0.35   # minimum cosine similarity to be included
TOP_K_DEFAULT = 10

# LangChain's RecursiveCharacterTextSplitter — splits on paragraph → sentence
# → word boundaries in order, preserving semantic coherence
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE * 4,    # ~4 chars per token
    chunk_overlap=CHUNK_OVERLAP * 4,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def _embed(texts: list[str]) -> list[list[float]]:
    """
    Get embeddings for a list of texts using text-embedding-3-small.
    Single API call regardless of list length — batched by OpenAI.
    Returns list of embedding vectors.
    """
    # OpenAI embedding API handles batches natively
    response = _openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors. Range: -1 to 1."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split all pages into chunks using RecursiveCharacterTextSplitter.
    Returns list of { text, url, chunk_index }.
    """
    all_chunks = []
    for page in pages:
        markdown = page.get("markdown", "").strip()
        url = page.get("url", "")
        if not markdown:
            continue

        texts = _splitter.split_text(markdown)
        for i, text in enumerate(texts):
            if text.strip():
                all_chunks.append({
                    "text": text.strip(),
                    "url": url,
                    "chunk_index": i,
                })
    return all_chunks


def extract_relevant_snippets(
    pages: list[dict],
    goal: str,
    top_k: int = TOP_K_DEFAULT,
) -> list[dict]:
    """
    Main entry point. Given scraped pages and a research goal string,
    returns the top-k most semantically relevant chunks.

    Steps:
      1. Chunk all pages (LangChain)
      2. Embed chunks + goal in one batched API call (OpenAI)
      3. Cosine similarity ranking
      4. Return top-k above threshold

    Each returned chunk: { text, url, chunk_index, relevance_score }
    Only these chunks go into Claude for synthesis — not full pages.
    """
    chunks = chunk_pages(pages)
    if not chunks:
        return []

    chunk_texts = [c["text"] for c in chunks]

    # Embed goal + all chunks in a single batched API call
    all_texts = [goal] + chunk_texts
    try:
        all_embeddings = _embed(all_texts)
    except Exception as e:
        print(f"[chunker] embedding error: {e}")
        # Fallback: return first top_k chunks without scoring
        return chunks[:top_k]

    goal_embedding = all_embeddings[0]
    chunk_embeddings = all_embeddings[1:]

    # Score each chunk
    for i, chunk in enumerate(chunks):
        chunk["relevance_score"] = _cosine_similarity(goal_embedding, chunk_embeddings[i])

    # Filter by threshold, sort by score
    relevant = [c for c in chunks if c["relevance_score"] >= RELEVANCE_THRESHOLD]
    relevant.sort(key=lambda c: c["relevance_score"], reverse=True)

    return relevant[:top_k]


def format_snippets_for_prompt(snippets: list[dict]) -> str:
    """
    Format snippets for injection into Claude synthesis prompts.
    Each snippet tagged with its source URL for claim mapping.
    """
    lines = []
    for i, s in enumerate(snippets):
        lines.append(f"[SOURCE {i+1}: {s['url']}]\n{s['text']}")
    return "\n\n---\n\n".join(lines)
