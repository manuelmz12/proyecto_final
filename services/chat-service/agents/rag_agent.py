import logging
import os

import chromadb
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

from bm25_retriever import bm25_search

logger = logging.getLogger(__name__)

COLLECTION_NAME = "arxiv_papers"
TOP_K = 5

SYSTEM_PROMPT = """You are a research assistant specializing in ArXiv scientific papers.
Answer the user's question based ONLY on the retrieved papers provided as context.
Always cite paper titles and authors when referencing specific work.
If the context doesn't contain enough information, say so explicitly.

Retrieved papers context:
{context}"""


def _get_chroma_client() -> chromadb.HttpClient:
    host = os.environ.get("CHROMA_HOST", "chroma")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    return chromadb.HttpClient(host=host, port=port)


def _embed_query(query: str) -> list[float]:
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS", "devlab-text-embedding-3-large")
    response = client.embeddings.create(input=query, model=model)
    return response.data[0].embedding


def _get_chat_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_MINI", "devlab-gpt-4o-mini"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        temperature=0.2,
    )


async def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Hybrid search: dense (ChromaDB) + lexical (BM25), deduplicated by title."""
    seen_titles: set[str] = set()
    results: list[dict] = []

    # ChromaDB
    try:
        query_embedding = _embed_query(query)
        client = _get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        dense_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            dense_results["documents"][0],
            dense_results["metadatas"][0],
            dense_results["distances"][0],
        ):
            title = meta.get("title", doc[:50])
            if title not in seen_titles:
                seen_titles.add(title)
                results.append({"document": doc, "metadata": meta, "score": 1 - dist})
    except Exception as e:
        logger.warning(f"Dense search failed: {e}")

    # BM25
    try:
        for item in bm25_search(query, top_k=top_k):
            if item["title"] not in seen_titles:
                seen_titles.add(item["title"])
                results.append({
                    "document": f"{item['title']}. {item['abstract']}",
                    "metadata": {
                        "title": item["title"],
                        "authors": ", ".join(item.get("authors", [])[:3]),
                        "published": item.get("published", ""),
                        "url": item.get("url", ""),
                    },
                    "score": item.get("score", 0.0),
                })
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")

    return results[:top_k]


async def rag_agent_run(query: str, chat_history: list[dict]) -> dict:
    """Execute hybrid RAG search and generate answer."""
    retrieved = await hybrid_search(query)

    if not retrieved:
        return {
            "answer": "I don't have enough information in the knowledge base to answer this question.",
            "sources": [],
            "retrieved_context": [],
        }

    context_parts = []
    for i, r in enumerate(retrieved, 1):
        meta = r["metadata"]
        context_parts.append(
            f"[{i}] Title: {meta.get('title', 'Unknown')}\n"
            f"    Authors: {meta.get('authors', 'Unknown')}\n"
            f"    Published: {meta.get('published', 'Unknown')}\n"
            f"    Content: {r['document'][:600]}"
        )
    context = "\n\n".join(context_parts)

    llm = _get_chat_llm()
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": query})

    response = await llm.ainvoke(messages)

    sources = [
        {"title": r["metadata"].get("title", ""), "url": r["metadata"].get("url", "")}
        for r in retrieved
    ]
    return {
        "answer": response.content,
        "sources": sources,
        "retrieved_context": [r["document"][:400] for r in retrieved],
    }
