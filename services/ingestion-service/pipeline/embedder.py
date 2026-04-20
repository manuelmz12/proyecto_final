import logging
import os
from typing import Any

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
)

EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS", "devlab-text-embedding-3-large")
BATCH_SIZE = 16


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


async def generate_embeddings(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate embeddings for paper title + abstract concatenation."""
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = _embed_batch(batch)
        all_embeddings.extend(embeddings)

    for paper, embedding in zip(papers, all_embeddings):
        paper["embedding"] = embedding

    logger.info(f"Generated embeddings for {len(papers)} papers")
    return papers
