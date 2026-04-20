import json
import logging
import os
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

INDEX_PATH = os.environ.get("BM25_INDEX_PATH", "/app/bm25_index/index.json")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def update_bm25_index(papers: list[dict[str, Any]]) -> None:
    """Append new papers to the persistent BM25 index."""
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    existing: list[dict] = []
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH) as f:
            existing = json.load(f)

    existing_ids = {p["arxiv_id"] for p in existing}
    new_papers = [p for p in papers if p["arxiv_id"] not in existing_ids]

    if not new_papers:
        logger.info("BM25 index: no new papers to add")
        return

    for p in new_papers:
        existing.append(
            {
                "arxiv_id": p["arxiv_id"],
                "title": p["title"],
                "abstract": p["abstract"],
                "authors": p["authors"],
                "published": p["published"],
                "url": p["url"],
            }
        )

    with open(INDEX_PATH, "w") as f:
        json.dump(existing, f)

    logger.info(f"BM25 index updated: {len(new_papers)} new papers, {len(existing)} total")


def load_bm25_index() -> tuple[BM25Okapi | None, list[dict]]:
    """Load BM25 index from disk. Returns (bm25, corpus_metadata)."""
    if not os.path.exists(INDEX_PATH):
        return None, []

    with open(INDEX_PATH) as f:
        corpus = json.load(f)

    tokenized = [_tokenize(f"{p['title']} {p['abstract']}") for p in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25, corpus


def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """Search BM25 index and return top_k results."""
    bm25, corpus = load_bm25_index()
    if bm25 is None or not corpus:
        return []

    scores = bm25.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"score": float(scores[i]), **corpus[i]} for i in top_indices if scores[i] > 0]
