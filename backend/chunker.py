"""
chunker.py — Text splitting, embedding, and relevance ranking.

This module is called inside tool implementations — never directly by agents.
The agent calls scrape_page(url, goal) and gets back relevant snippets.
The chunking and embedding happens invisibly inside the tool.

Pipeline:
  1. Split markdown via RecursiveCharacterTextSplitter (LangChain)
  2. Embed all chunks + the goal in one batched OpenAI API call
  3. Cosine similarity between goal vector and each chunk vector
  4. Return top-k chunks above relevance threshold

Why embedding model and not Claude:
  Embedding models are purpose-built for semantic similarity.
  They are not LLMs — they produce dense vectors, not text.
  Cost is ~$0.00002 per 1K tokens vs ~$0.003 for Haiku.
  Claude is reserved for reasoning only.
"""

import os
import numpy as np
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL  = "text-embedding-3-small"
CHUNK_SIZE       = 400   # tokens (~1600 chars)
CHUNK_OVERLAP    = 50    # token overlap to avoid cutting mid-sentence
RELEVANCE_THRESHOLD = 0.3
TOP_K            = 8

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE * 4,
    chunk_overlap=CHUNK_OVERLAP * 4,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def _embed(texts: list[str]) -> list[list[float]]:
    """Batch embed a list of strings. Single API call regardless of list size."""
    response = _openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def get_relevant_snippets(markdown: str, url: str, goal: str) -> list[dict]:
    """
    Given a page's markdown, a source URL, and a research goal:
    1. Split into chunks
    2. Embed chunks + goal together in one API call
    3. Rank by cosine similarity to goal
    4. Return top-k above threshold

    Each returned snippet: { text, url, score }
    """
    chunks = _splitter.split_text(markdown)
    if not chunks:
        return []

    # Embed goal + all chunks in a single batched call
    all_texts  = [goal] + chunks
    embeddings = _embed(all_texts)

    goal_vec    = embeddings[0]
    chunk_vecs  = embeddings[1:]

    scored = []
    for i, chunk in enumerate(chunks):
        score = _cosine_similarity(goal_vec, chunk_vecs[i])
        if score >= RELEVANCE_THRESHOLD:
            scored.append({"text": chunk, "url": url, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:TOP_K]
