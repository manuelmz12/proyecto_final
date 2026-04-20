import json
import logging
import os
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
HISTORY_TTL = 3600
MAX_MESSAGES = 20

_redis_pool: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_pool


async def get_conversation_history(session_id: str) -> list[dict[str, str]]:
    r = get_redis()
    key = f"conversation:{session_id}"
    raw = await r.get(key)
    if raw:
        return json.loads(raw)
    return []


async def append_message(session_id: str, role: str, content: str) -> None:
    r = get_redis()
    key = f"conversation:{session_id}"
    history = await get_conversation_history(session_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_MESSAGES:
        history = history[-MAX_MESSAGES:]
    await r.set(key, json.dumps(history), ex=HISTORY_TTL)


async def clear_conversation(session_id: str) -> None:
    r = get_redis()
    await r.delete(f"conversation:{session_id}")
