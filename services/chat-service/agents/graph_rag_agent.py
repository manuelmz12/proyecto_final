import json
import logging
import os

from langchain_openai import AzureChatOpenAI
from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

CYPHER_GENERATION_PROMPT = """You are an expert at generating Cypher queries for a Neo4j graph database about ArXiv research papers.

Graph schema:
- (Paper {arxiv_id, title, abstract, published, url, categories})
- (Author {name})
- (Institution {name})
- (Concept {name})
- (Paper)-[:CITES]->(Paper)
- (Author)-[:WROTE]->(Paper)
- (Paper)-[:FROM_INSTITUTION]->(Institution)
- (Paper)-[:COVERS]->(Concept)

Generate a Cypher READ query to answer the user's question.
Return ONLY valid JSON: {"cypher": "<query>", "explanation": "<brief explanation>"}

Rules:
- Use LIMIT 10 unless user asks for more
- Use case-insensitive matching with toLower() for name searches
- Never generate WRITE queries (CREATE, MERGE, DELETE, SET)
- If the question cannot be answered with the graph, set cypher to null

User question: {question}"""

SYNTHESIS_PROMPT = """You are a research assistant. Based on the graph database query results below,
answer the user's question in a clear, informative way. Cite specific paper titles and author names.

Graph query results:
{results}

User question: {question}"""


def _get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_CHAT", "devlab-gpt-4o-mini"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        temperature=0,
    )


async def _generate_cypher(question: str) -> dict:
    llm = _get_llm()
    messages = [{"role": "user", "content": CYPHER_GENERATION_PROMPT.format(question=question)}]
    response = await llm.ainvoke(messages)
    return json.loads(response.content)


async def _execute_cypher(cypher: str) -> list[dict]:
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        async with driver.session() as session:
            result = await session.run(cypher)
            records = await result.data()
            return records[:20]
    finally:
        await driver.close()


async def graph_rag_agent_run(query: str) -> dict:
    """Generate Cypher from query, execute it, and synthesize a natural language answer."""
    try:
        cypher_result = await _generate_cypher(query)
        cypher = cypher_result.get("cypher")

        if not cypher:
            return {
                "answer": "I couldn't generate a graph query for this question. Try rephrasing or use the RAG agent.",
                "cypher_used": None,
                "graph_results": [],
            }

        logger.info(f"Executing Cypher: {cypher}")
        records = await _execute_cypher(cypher)

        if not records:
            return {
                "answer": "No matching data found in the knowledge graph for your query.",
                "cypher_used": cypher,
                "graph_results": [],
            }

        results_str = json.dumps(records, indent=2, default=str)
        llm = _get_llm()
        messages = [
            {"role": "user", "content": SYNTHESIS_PROMPT.format(results=results_str, question=query)}
        ]
        response = await llm.ainvoke(messages)

        return {
            "answer": response.content,
            "cypher_used": cypher,
            "graph_results": records,
        }

    except json.JSONDecodeError:
        return {"answer": "Failed to generate a valid graph query.", "cypher_used": None, "graph_results": []}
    except Exception as e:
        logger.error(f"GraphRAG agent failed: {e}", exc_info=True)
        return {"answer": f"Graph query error: {str(e)}", "cypher_used": None, "graph_results": []}
