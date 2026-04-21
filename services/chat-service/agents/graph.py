"""
LangGraph multi-agent workflow for Omni-Analyst.

Flow:
  input → router → [rag_node, graph_node] (parallel or single) → synthesis → output
"""

import asyncio
import logging
import os
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

try:
    from langfuse.callback import CallbackHandler as LangfuseCallback
    _langfuse_available = (
        bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and
        bool(os.environ.get("LANGFUSE_SECRET_KEY"))
    )
except ImportError:
    _langfuse_available = False

from agents.router_agent import route_query
from agents.rag_agent import rag_agent_run
from agents.graph_rag_agent import graph_rag_agent_run
from agents.synthesis_agent import synthesis_agent_run

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    query: str
    chat_history: list[dict]
    intent: str
    rag_result: dict
    graph_result: dict
    final_answer: str
    sources: list[dict]
    cypher_used: str | None


async def router_node(state: AgentState) -> AgentState:
    result = await route_query(state["query"])
    return {**state, "intent": result["intent"]}


async def rag_node(state: AgentState) -> AgentState:
    if state["intent"] not in ("rag", "hybrid"):
        return {**state, "rag_result": {"answer": "", "sources": [], "retrieved_context": []}}
    result = await rag_agent_run(state["query"], state["chat_history"])
    return {**state, "rag_result": result}


async def graph_node(state: AgentState) -> AgentState:
    if state["intent"] not in ("graph", "hybrid"):
        return {**state, "graph_result": {"answer": "", "cypher_used": None, "graph_results": []}}
    result = await graph_rag_agent_run(state["query"])
    return {**state, "graph_result": result, "cypher_used": result.get("cypher_used")}


async def parallel_retrieval_node(state: AgentState) -> AgentState:
    """Run RAG and GraphRAG in parallel for hybrid queries."""
    rag_task = asyncio.create_task(rag_agent_run(state["query"], state["chat_history"]))
    graph_task = asyncio.create_task(graph_rag_agent_run(state["query"]))
    rag_result, graph_result = await asyncio.gather(rag_task, graph_task)
    return {
        **state,
        "rag_result": rag_result,
        "graph_result": graph_result,
        "cypher_used": graph_result.get("cypher_used"),
    }


async def synthesis_node(state: AgentState) -> AgentState:
    answer = await synthesis_agent_run(
        query=state["query"],
        rag_result=state["rag_result"],
        graph_result=state["graph_result"],
        chat_history=state["chat_history"],
    )
    sources = state["rag_result"].get("sources", [])
    return {**state, "final_answer": answer, "sources": sources}


async def general_node(state: AgentState) -> AgentState:
    """Handle out-of-scope or meta questions directly."""
    if any(greet in state["query"].lower() for greet in ["hello", "hi", "hola", "hey"]):
        answer = (
            "Hello! I'm Omni-Analyst, your academic research assistant for ArXiv papers. "
            "I can help you:\n"
            "- Find and summarize papers on any ML/AI topic\n"
            "- Explore author networks and citation relationships\n"
            "- Answer questions about research methodologies\n\n"
            "What would you like to know?"
        )
    else:
        answer = (
            "I'm specialized in academic research papers from ArXiv. "
            "I can answer questions about ML, AI, NLP papers, their authors, and research relationships. "
            "Please ask me something related to scientific literature."
        )
    return {**state, "final_answer": answer, "sources": []}


def route_after_router(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "rag":
        return "rag"
    elif intent == "graph":
        return "graph"
    elif intent == "hybrid":
        return "parallel"
    else:
        return "general"


def build_agent_graph() -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("graph", graph_node)
    workflow.add_node("parallel", parallel_retrieval_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("general", general_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {"rag": "rag", "graph": "graph", "parallel": "parallel", "general": "general"},
    )
    workflow.add_edge("rag", "synthesis")
    workflow.add_edge("graph", "synthesis")
    workflow.add_edge("parallel", "synthesis")
    workflow.add_edge("synthesis", END)
    workflow.add_edge("general", END)

    return workflow.compile()


# Singleton compiled graph
_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph


async def run_agent(query: str, chat_history: list[dict], session_id: str) -> dict:
    """Run the full multi-agent pipeline."""
    graph = get_compiled_graph()
    initial_state: AgentState = {
        "query": query,
        "chat_history": chat_history,
        "intent": "",
        "rag_result": {},
        "graph_result": {},
        "final_answer": "",
        "sources": [],
        "cypher_used": None,
    }
    logger.info(f"Running agent for session={session_id}, query='{query[:60]}...'")
    config = {}
    if _langfuse_available:
        handler = LangfuseCallback(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "http://langfuse:3000"),
            session_id=session_id,
            user_id=session_id,
        )
        config = {"callbacks": [handler]}
    result = await graph.ainvoke(initial_state, config=config)
    return {
        "answer": result["final_answer"],
        "intent": result["intent"],
        "sources": result["sources"],
        "cypher_used": result.get("cypher_used"),
    }
