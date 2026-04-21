import json
import logging
import os

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

INDEX_PATH = os.environ.get("BM25_INDEX_PATH", "/app/bm25_index/index.json")


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """Search BM25 index and return top_k results."""
    if not os.path.exists(INDEX_PATH):
        return []
    with open(INDEX_PATH) as f:
        corpus = json.load(f)
    if not corpus:
        return []
    tokenized = [_tokenize(f"{p['title']} {p['abstract']}") for p in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"score": float(scores[i]), **corpus[i]} for i in top_indices if scores[i] > 0]
