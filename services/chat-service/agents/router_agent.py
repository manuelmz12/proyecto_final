import json
import logging
import os

from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a query router for an academic research intelligence system about ArXiv papers.
Analyze the user's query and classify it into one of these categories:

- "rag": Questions about paper content, summaries, methodologies, technical explanations.
  Examples: "What is RAG?", "Explain attention mechanisms", "What papers discuss transformers?"

- "graph": Questions about relationships between entities: authors, institutions, citations, collaborations.
  Examples: "Who are the authors of paper X?", "What institution is author Y from?",
  "What papers cite concept Z?", "Who collaborates with researcher W?"

- "hybrid": Questions that need both content search AND relationship queries.
  Examples: "Find influential papers on diffusion models and their authors",
  "Which MIT researchers published about LLMs in 2024?"

- "general": Greetings, meta-questions about the system, questions out of scope.
  Examples: "Hello", "What can you do?", "What is the weather?"

Respond ONLY with valid JSON: {"intent": "<category>", "reasoning": "<brief explanation>"}"""


def get_router_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_MINI", "devlab-gpt-4o-mini"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        temperature=0,
    )


async def route_query(query: str) -> dict:
    """Classify the user query intent."""
    llm = get_router_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = await llm.ainvoke(messages)
    try:
        result = json.loads(response.content)
        intent = result.get("intent", "general")
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        intent = "general"
        reasoning = "Failed to parse router response"

    logger.info(f"Router: '{query[:50]}...' → intent='{intent}', reason='{reasoning}'")
    return {"intent": intent, "reasoning": reasoning}
