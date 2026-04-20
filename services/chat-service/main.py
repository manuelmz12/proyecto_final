import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.responses import Response

from agents.graph import run_agent
from auth.jwt_handler import get_current_user, require_admin
from guardrails.input_filter import validate_input
from memory.redis_memory import append_message, clear_conversation, get_conversation_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHAT_REQUESTS = Counter("chat_requests_total", "Total chat requests", ["status"])
CHAT_LATENCY = Histogram("chat_latency_seconds", "Chat request latency", buckets=[0.5, 1, 2, 5, 10, 30])
GUARDRAIL_BLOCKS = Counter("guardrail_blocks_total", "Total guardrail blocks", ["stage"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Chat service started")
    yield
    logger.info("Chat service stopped")


app = FastAPI(title="Omni-Analyst Chat Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    intent: str
    sources: list[dict]
    cypher_used: str | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    session_id = request.session_id or str(uuid.uuid4())

    # Input guardrail
    input_check = validate_input(request.message)
    if not input_check["safe"]:
        GUARDRAIL_BLOCKS.labels(stage="input").inc()
        CHAT_REQUESTS.labels(status="blocked_input").inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input blocked: {input_check['reason']}",
        )

    sanitized_query = input_check["text"]
    history = await get_conversation_history(session_id)

    try:
        with CHAT_LATENCY.time():
            result = await run_agent(sanitized_query, history, session_id)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        CHAT_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Agent processing failed")

    final_answer = result["answer"]
    await append_message(session_id, "user", sanitized_query)
    await append_message(session_id, "assistant", final_answer)

    CHAT_REQUESTS.labels(status="success").inc()
    return ChatResponse(
        answer=final_answer,
        session_id=session_id,
        intent=result["intent"],
        sources=result["sources"],
        cypher_used=result.get("cypher_used"),
    )


@app.delete("/chat/{session_id}/history")
async def clear_history(session_id: str, current_user: dict = Depends(get_current_user)):
    await clear_conversation(session_id)
    return {"message": f"History cleared for session {session_id}"}


@app.get("/auth/verify")
async def verify_credentials(current_user: dict = Depends(get_current_user)):
    """Endpoint ligero para que el frontend verifique credenciales en el login."""
    return {"username": current_user["username"], "role": current_user["role"]}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chat-service"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/catalog")
async def get_catalog(current_user: dict = Depends(get_current_user)):
    import yaml
    with open("catalog/agents_catalog.yaml") as f:
        return yaml.safe_load(f)
