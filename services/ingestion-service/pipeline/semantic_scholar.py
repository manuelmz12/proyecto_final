import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper"
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# Without API key: ~100 req/5min. With key: 1 req/s.
# Semaphore limits concurrent requests; delay avoids burst rate limits.
_CONCURRENCY = 3 if API_KEY else 1
_DELAY = 1.0 if API_KEY else 3.5
_MAX_RETRIES = 3


async def _fetch_paper_data(
    client: httpx.AsyncClient, arxiv_id: str, semaphore: asyncio.Semaphore
) -> dict | None:
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    url = f"{SEMANTIC_SCHOLAR_BASE}/arXiv:{arxiv_id}"
    params = {"fields": "citations.title,citations.year,authors.affiliations,influentialCitationCount"}

    async with semaphore:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = _DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited for {arxiv_id}, waiting {wait:.1f}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                elif resp.status_code == 404:
                    return None
                else:
                    logger.warning(f"Semantic Scholar {resp.status_code} for {arxiv_id}")
                    return None
            except Exception as e:
                logger.warning(f"Semantic Scholar fetch failed for {arxiv_id}: {e}")
                return None

        logger.warning(f"Giving up on {arxiv_id} after {_MAX_RETRIES} attempts")
        return None


async def enrich_with_citations(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich papers with citation and affiliation data from Semantic Scholar."""
    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [_fetch_paper_data(client, p["arxiv_id"], semaphore) for p in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    enriched = 0
    for paper, result in zip(papers, results):
        if isinstance(result, dict) and result:
            citations = result.get("citations", [])
            paper["citations"] = [c.get("title", "") for c in citations[:10] if c.get("title")]
            paper["influential_citation_count"] = result.get("influentialCitationCount", 0)

            all_affiliations = []
            for author_data in result.get("authors", []):
                for aff in author_data.get("affiliations", []):
                    if aff and aff not in all_affiliations:
                        all_affiliations.append(aff)
            paper["institutions"] = all_affiliations[:5]
            enriched += 1

    logger.info(f"Enriched {enriched}/{len(papers)} papers with Semantic Scholar data")
    return papers
