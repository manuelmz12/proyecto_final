import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx

logger = logging.getLogger(__name__)

ARXIV_BASE_URL = "https://export.arxiv.org/api/query"


async def fetch_arxiv_papers(
    categories: list[str], max_papers: int, days_back: int
) -> list[dict[str, Any]]:
    """Fetch papers from ArXiv API for given categories."""
    query = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_papers,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(ARXIV_BASE_URL, params=params)
        resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    papers = []

    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if published < cutoff:
            continue

        arxiv_id = entry.id.split("/abs/")[-1]
        authors = [a.name for a in entry.get("authors", [])]
        categories_list = [tag.term for tag in entry.get("tags", [])]

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": entry.title.replace("\n", " ").strip(),
                "abstract": entry.summary.replace("\n", " ").strip(),
                "authors": authors,
                "categories": categories_list,
                "published": published.isoformat(),
                "url": entry.link,
                "citations": [],
                "institutions": [],
            }
        )

    logger.info(f"Fetched {len(papers)} papers from ArXiv")
    return papers
