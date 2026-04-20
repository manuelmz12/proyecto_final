import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper"
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


async def _fetch_paper_data(client: httpx.AsyncClient, arxiv_id: str) -> dict | None:
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    url = f"{SEMANTIC_SCHOLAR_BASE}/arXiv:{arxiv_id}"
    params = {"fields": "citations.title,citations.year,authors.affiliations,influentialCitationCount"}
    try:
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            await asyncio.sleep(2)
        return None
    except Exception as e:
        logger.warning(f"Semantic Scholar fetch failed for {arxiv_id}: {e}")
        return None


async def enrich_with_citations(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich papers with citation and affiliation data from Semantic Scholar."""
    async with httpx.AsyncClient(timeout=20) as client:
        tasks = [_fetch_paper_data(client, p["arxiv_id"]) for p in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for paper, result in zip(papers, results):
        if isinstance(result, dict) and result:
            citations = result.get("citations", [])
            paper["citations"] = [c.get("title", "") for c in citations[:10]]
            paper["influential_citation_count"] = result.get("influentialCitationCount", 0)

            all_affiliations = []
            for author_data in result.get("authors", []):
                for aff in author_data.get("affiliations", []):
                    if aff and aff not in all_affiliations:
                        all_affiliations.append(aff)
            paper["institutions"] = all_affiliations[:5]

    logger.info(f"Enriched {len(papers)} papers with Semantic Scholar data")
    return papers
