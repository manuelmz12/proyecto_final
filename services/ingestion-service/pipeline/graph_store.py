import json
import logging
import os
from typing import Any

from neo4j import AsyncGraphDatabase
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

llm_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
)
MINI_MODEL = os.environ.get("AZURE_OPENAI_DEPLOYMENT_MINI", "devlab-gpt-4o-mini")

ENTITY_EXTRACTION_PROMPT = """Extract structured entities from this research paper abstract.
Return ONLY valid JSON with this exact schema:
{
  "concepts": ["list of key technical concepts/methods, max 5"],
  "institutions": ["list of research institutions mentioned, max 5"]
}

Paper title: {title}
Abstract: {abstract}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _extract_entities(title: str, abstract: str) -> dict:
    response = llm_client.chat.completions.create(
        model=MINI_MODEL,
        messages=[
            {"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(title=title, abstract=abstract[:800])}
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


async def upsert_to_neo4j(papers: list[dict[str, Any]]) -> None:
    """Build graph of papers, authors, institutions, and concepts in Neo4j."""
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    async with driver.session() as session:
        await session.run("""
            CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE
        """)
        await session.run("""
            CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE
        """)
        await session.run("""
            CREATE CONSTRAINT institution_name IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE
        """)
        await session.run("""
            CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE
        """)

        for paper in papers:
            entities = {"concepts": [], "institutions": paper.get("institutions", [])}
            try:
                extracted = _extract_entities(paper["title"], paper["abstract"])
                entities["concepts"] = extracted.get("concepts", [])
                if not entities["institutions"]:
                    entities["institutions"] = extracted.get("institutions", [])
            except Exception as e:
                logger.warning(f"Entity extraction failed for {paper['arxiv_id']}: {e}")

            await session.run(
                """
                MERGE (p:Paper {arxiv_id: $arxiv_id})
                SET p.title = $title,
                    p.abstract = $abstract,
                    p.published = $published,
                    p.url = $url,
                    p.categories = $categories
                """,
                arxiv_id=paper["arxiv_id"],
                title=paper["title"],
                abstract=paper["abstract"][:500],
                published=paper["published"],
                url=paper["url"],
                categories=paper["categories"],
            )

            for author in paper["authors"][:5]:
                await session.run(
                    """
                    MERGE (a:Author {name: $name})
                    WITH a
                    MATCH (p:Paper {arxiv_id: $arxiv_id})
                    MERGE (a)-[:WROTE]->(p)
                    """,
                    name=author,
                    arxiv_id=paper["arxiv_id"],
                )

            for institution in entities["institutions"]:
                if institution:
                    await session.run(
                        """
                        MERGE (i:Institution {name: $name})
                        WITH i
                        MATCH (p:Paper {arxiv_id: $arxiv_id})
                        MERGE (p)-[:FROM_INSTITUTION]->(i)
                        """,
                        name=institution,
                        arxiv_id=paper["arxiv_id"],
                    )

            for concept in entities["concepts"]:
                if concept:
                    await session.run(
                        """
                        MERGE (c:Concept {name: $name})
                        WITH c
                        MATCH (p:Paper {arxiv_id: $arxiv_id})
                        MERGE (p)-[:COVERS]->(c)
                        """,
                        name=concept,
                        arxiv_id=paper["arxiv_id"],
                    )

            for cited_title in paper.get("citations", []):
                if cited_title:
                    await session.run(
                        """
                        MERGE (cited:Paper {title: $cited_title})
                        WITH cited
                        MATCH (p:Paper {arxiv_id: $arxiv_id})
                        MERGE (p)-[:CITES]->(cited)
                        """,
                        cited_title=cited_title,
                        arxiv_id=paper["arxiv_id"],
                    )

    await driver.close()
    logger.info(f"Upserted {len(papers)} papers to Neo4j")
