import logging
import os

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Omni-Analyst, an intelligent research assistant specialized in ArXiv scientific papers.
Your role is to synthesize information from multiple sources and provide comprehensive, accurate answers.

Guidelines:
- Synthesize all provided context into a coherent, well-structured response
- Always cite paper titles and authors when referencing specific work
- Be concise but thorough; use bullet points or numbered lists when helpful
- If information is missing or uncertain, acknowledge it explicitly

Context from RAG search:
{rag_context}

Context from Knowledge Graph:
{graph_context}"""


def _get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_CHAT", "devlab-gpt-4o-mini"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        temperature=0.3,
    )


async def synthesis_agent_run(
    query: str,
    rag_result: dict,
    graph_result: dict,
    chat_history: list[dict],
) -> str:
    """Consolidate RAG + GraphRAG results and generate final answer."""
    rag_context = rag_result.get("answer", "No RAG results available.")
    graph_context = graph_result.get("answer", "No graph results available.")

    llm = _get_llm()
    system_content = SYSTEM_PROMPT.format(rag_context=rag_context, graph_context=graph_context)

    messages = [SystemMessage(content=system_content)]
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))

    response = await llm.ainvoke(messages)
    return response.content
