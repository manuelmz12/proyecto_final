import logging
import os
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
COLLECTION_NAME = "arxiv_papers"


def _get_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


async def upsert_to_chroma(papers: list[dict[str, Any]]) -> None:
    """Upsert paper embeddings and metadata into ChromaDB."""
    client = _get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [p["arxiv_id"] for p in papers]
    embeddings = [p["embedding"] for p in papers]
    documents = [f"{p['title']}. {p['abstract']}" for p in papers]
    metadatas = [
        {
            "title": p["title"],
            "authors": ", ".join(p["authors"][:5]),
            "categories": ", ".join(p["categories"][:5]),
            "published": p["published"],
            "url": p["url"],
            "influential_citations": p.get("influential_citation_count", 0),
        }
        for p in papers
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info(f"Upserted {len(papers)} papers to ChromaDB collection '{COLLECTION_NAME}'")
